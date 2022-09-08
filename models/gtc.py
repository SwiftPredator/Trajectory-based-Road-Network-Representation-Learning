import math
import os
from ast import walk
from operator import itemgetter
from turtle import forward

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from _walker import random_walks as _random_walks
from scipy import sparse
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GAE, GATConv, GCNConv, InnerProductDecoder, SGConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import (
    add_self_loops,
    from_networkx,
    negative_sampling,
    remove_self_loops,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from tqdm import tqdm

from .model import Model
from .utils import generate_trajid_to_nodeid, transform_data


class GTCModel(Model):
    def __init__(
        self,
        data,
        device,
        network,
        traj_data=None,
        emb_dim: str = 128,
        adj=None,
        k: int = 1,
        bidirectional=True,
        add_self_loops=True,
        norm=False,
    ):
        self.model = GTCSubConv(data.x.shape[1], emb_dim, norm=norm)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model = self.model.to(device)
        self.device = device
        self.network = network
        if adj is None and traj_data is not None:
            self.traj_to_node = generate_trajid_to_nodeid(network)
            self.traj_data = traj_data["seg_seq"].tolist()
            adj = self.generate_node_traj_adj(
                k=k, bidirectional=bidirectional, add_self_loops=add_self_loops
            )
            np.savetxt(
                "./traj_adj_k_" + str(k) + "_" + str(bidirectional) + ".gz", X=adj
            )
        # else:
        # adj = np.loadtxt(load_traj_adj_path)
        self.train_data = transform_data(data, adj)
        self.train_data = self.train_data.to(device)

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model(
                self.train_data.x,
                self.train_data.edge_traj_index,
                self.train_data.edge_weight,
            )
            loss = self.recon_loss(z, self.train_data.edge_traj_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 1000 == 0:
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

    def generate_node_traj_adj(
        self, k: int = np.inf, bidirectional=True, add_self_loops=True
    ):
        nodes = list(self.network.line_graph.nodes)
        adj = nx.to_numpy_array(self.network.line_graph)
        np.fill_diagonal(adj, 0)

        if add_self_loops:
            adj += np.eye(len(nodes), len(nodes))

        for traj in tqdm(self.traj_data):
            # print(traj)
            for i, traj_node in enumerate(traj):
                if k == -1:
                    k = len(traj)
                left_slice, right_slice = min(k, i) if bidirectional else 0, min(
                    k + 1, len(traj) - i
                )
                traj_nodes = traj[(i - left_slice) : (i + right_slice)]
                # convert traj_nodes to graph_nodes
                target = itemgetter(traj_node)(self.traj_to_node)
                context = itemgetter(*traj_nodes)(self.traj_to_node)
                adj[target, context] += 1
        # remove self weighting if no self loops should be allowed
        if not add_self_loops:
            np.fill_diagonal(adj, 0)
            zero_rows = np.where(~adj.any(axis=1))[0]
            for idx in zero_rows:
                adj[idx, idx] = 1

        # norm adj row wise to get probs
        rowsum = adj.sum(axis=1, keepdims=True)
        adj = adj / rowsum

        # convert to edge_index

        return adj

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
            self.model(
                self.train_data.x,
                self.train_data.edge_traj_index,
                self.train_data.edge_weight,
            )
            .detach()
            .cpu()
            .numpy()
        )


class GTCSubConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm=False):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.norm_layer = LayerNorm(out_dim)
        self.norm = norm

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x.float(), edge_index, edge_weight)
        if self.norm:
            x = self.norm_layer(x)
        return x


class GTNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        ...
