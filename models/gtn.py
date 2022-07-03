import os
from operator import itemgetter
from turtle import forward

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, GATConv, GCNConv
from torch_geometric.nn.conv import MessagePassing

from .model import Model


class GTNModel(Model):
    def __init__(self, data, device, network, emb_dim=128):
        self.model = (
            ...
        )  # GAE(encoder(data.x.shape[1], emb_dim))  # feature dim, emb dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model = self.model.to(device)
        self.device = device
        self.train_data = data
        self.train_data = self.train_data.to(device)

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model(self.train_data.x, self.train_data.edge_index)
            loss = self.model.recon_loss(z, self.train_data.edge_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 500 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

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
    def __init__(self, traj_data: pd.DataFrame, network):
        super().__init__()
        self.traj_data = traj_data["cpath"].tolist()
        self.network = network
        self.node_to_traj = self.generate_nodeid_to_trajid()
        self.traj_to_node = self.generate_trajid_to_nodeid()

    def forward(self):
        ...

    def generate_node_traj_adj(self, k: int = np.inf):
        nodes = list(self.network.line_graph.nodes)
        adj = np.zeros(shape=(len(nodes), len(nodes)))
        for traj in self.traj_data:
            for i, traj_node in enumerate(traj):
                left_slice, right_slice = min(k, i), min(k, len(traj) - i)
                traj_nodes = traj[(i - left_slice) : (i + right_slice)]
                # convert traj_nodes to graph_nodes
                graph_nodes = itemgetter(traj_nodes)(map)
                adj[graph_nodes[i], graph_nodes] += 1

            break

    def generate_nodeid_to_trajid(self):
        ...

    def generate_trajid_to_nodeid(self):
        map = {}
        nodes = list(self.network.line_graph.nodes)
        for index, id in zip(self.network.gdf_edges.index, self.network.gdf_edges.fid):
            map[id] = nodes.index(index)

        return map


class GTNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        ...
