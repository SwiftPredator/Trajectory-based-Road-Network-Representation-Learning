import os
from operator import itemgetter
from turtle import forward

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, GATConv, GCNConv, InnerProductDecoder
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    add_self_loops,
    from_networkx,
    negative_sampling,
    remove_self_loops,
)
from tqdm import tqdm

from .model import Model


class GTNModel(Model):
    def __init__(self, data, device, network, traj_data, emb_dim=128):
        self.model = GTNSubConv(data.x.shape[1], emb_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model = self.model.to(device)
        self.device = device
        self.traj_data = traj_data["seg_seq"].tolist()
        self.network = network
        self.traj_to_node = self.generate_trajid_to_nodeid()

        self.train_data = self.transform_data(data)
        self.train_data = self.train_data.to(device)

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model(
                self.train_data.x,
                self.train_data.edge_index,
                self.train_data.edge_weight,
            )
            loss = self.recon_loss(z, self.train_data.edge_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 500 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        decoder = InnerProductDecoder()
        EPS = 1e-15

        pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def transform_data(self, data):
        adj = self.generate_node_traj_adj(k=1)
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
        data.edge_index = from_networkx(G).edge_index
        data.edge_weight = torch.Tensor(
            list(nx.get_edge_attributes(G, "weight").values())
        )

        return data

    def generate_node_traj_adj(self, k: int = np.inf):
        nodes = list(self.network.line_graph.nodes)
        adj = np.eye(len(nodes), len(nodes))  # adj with initial self loops
        for traj in tqdm(self.traj_data):
            # print(traj)
            for i, traj_node in enumerate(traj):
                left_slice, right_slice = min(k, i), min(k + 1, len(traj) - i)
                traj_nodes = traj[(i - left_slice) : (i + right_slice)]
                # convert traj_nodes to graph_nodes
                target = itemgetter(traj_node)(self.traj_to_node)
                context = itemgetter(*traj_nodes)(self.traj_to_node)
                adj[target, context] += 1

        # norm adj row wise
        rowsum = adj.sum(axis=1, keepdims=True)
        adj = adj / rowsum

        # convert to edge_index

        return adj

    def generate_trajid_to_nodeid(self):
        map = {}
        nodes = list(self.network.line_graph.nodes)
        for index, id in zip(self.network.gdf_edges.index, self.network.gdf_edges.fid):
            map[id] = nodes.index(index)

        return map

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), os.path.join(path + "/model.pt"))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        np.savetxt(
            os.path.join(path + "/embedding.out"),
            X=self.model.encode(self.train_data.x, self.train_data.edge_index)
            .detach()
            .cpu()
            .numpy(),
        )

    def load_emb(self, path=None):
        if path:
            return np.loadtxt(path)
        return (
            self.model.encode(self.train_data.x, self.train_data.edge_index)
            .detach()
            .cpu()
        )


class GTNSubConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x.float(), edge_index, edge_weight)
        return x


class GTNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        ...
