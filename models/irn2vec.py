import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse.csgraph import shortest_path

from .model import Model


class IRN2VecModel(Model):
    def __init__(
        self,
        data,
        device,
        graph: nx.Graph,
        emb_dim: int = 128,
        n_shortest_paths: int = 1250,
    ):
        """
        Args:
            data (_type_): placeholder
            device (_type_): torch device
            network (nx.Graph): graph of city where nodes are intersections and edges are roads
            emb_dim (int, optional): embedding dimension. Defaults to 128.
        """
        self.device = device
        self.emb_dim = emb_dim
        self.data = self.preprocess_dataset(graph, n_shortest_paths)
        self.model = IRN2Vec()

    def preprocess_dataset(self, G, n_shortest_paths: int):
        print("preprocess")
        adj = nx.adjacency_matrix(G)

        def get_path(Pr, i, j):
            path = [j]
            k = j
            while Pr[i, k] != -9999:
                path.append(Pr[i, k])
                k = Pr[i, k]
            if Pr[i, k] == -9999 and k != i:
                return []

            return path[::-1]

        _, P = shortest_path(
            adj, directed=True, method="D", return_predecessors=True, unweighted=True
        )

        nodes = np.arange(len(G.nodes))
        paths = []
        for source in nodes:
            targets = np.random.choice(
                np.setdiff1d(np.arange(len(G.nodes)), [source]), size=n_shortest_paths
            )
            node_paths = []
            for target in targets:
                node_paths.append(get_path(P, source, target))
            paths.append(node_paths)

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model.encode(self.train_data.x, self.train_data.edge_index)
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
            .numpy()
        )


class IRN2Vec(nn.Module):
    def __init__(self, graph, emb_dim):
        self.lin_vx = nn.Linear(len(graph.nodes), emb_dim)
        self.lin_vy = nn.Linear(len(graph.nodes), emb_dim)

        self.lin_out = nn.Linear(emb_dim, 3)
        self.act_out = nn.Sigmoid()

    def forward(self, vx, vy):
        x = self.lin_vx(vx)
        y = self.lin_vy(vy)

        x = x * y  # aggregate embeddings

        x = self.lin_out(x)

        return self.act_out(x)
