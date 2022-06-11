import os
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

from .model import Model


class SRN2VecModel(Model):
    def __init__(
        self,
        data,
        device,
        network,
        emb_dim: int = 128,
        n_shortest_paths: int = 1280,
        window_size: int = 900,  # 50 meters
        number_negative: int = 7,
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
        self.data = self.preprocess_dataset(
            network, n_shortest_paths, window_size, number_negative
        )
        self.model = SRN2Vec(network, emb_dim)

    def preprocess_dataset(
        self, network, n_shortest_paths: int, window_size: int, number_negative: int
    ):
        paths = self.generate_shortest_paths(
            network.line_graph, n_shortest_paths=n_shortest_paths
        )
        trainset = self.generate_train_pairs(
            network, paths, window_size, number_negative
        )

        return trainset

    def generate_train_pairs(
        self, network, paths: list, window_size: int, number_negative: int
    ):
        # generate windows and extract features from paper (same inter type and same degree)
        train_pairs = []
        info = network.gdf_edges
        node_list = list(network.line_graph.nodes)
        for node_paths in tqdm(paths):
            for node_path in node_paths:
                for i in range(len(node_path)):
                    window = [node_path[i]]
                    # print(node_path[i])
                    current_length = info.loc[node_path[i], "length"]
                    for j in range(i, len(node_path) - 1):
                        edge = info.loc[node_path[j + 1]]
                        if current_length + edge["length"] <= window_size:
                            window.append(node_path[j + 1])
                            current_length += edge["length"]
                        else:
                            break
                    if len(window) > 1:
                        combs = combinations(window, 2)
                        for comb in combs:
                            node1 = info.loc[comb[0]]
                            node2 = info.loc[comb[1]]
                            same_type = node1["highway_enc"] == node2["highway_enc"]
                            index1 = node_list.index(comb[0])
                            index2 = node_list.index(comb[1])
                            # generate positive sample
                            train_pairs.append(
                                (
                                    int(index1),
                                    int(index2),
                                    1,
                                    int(same_type),
                                )
                            )
                            # generate negative sample
                            for _ in range(number_negative):
                                neg_v = np.random.choice(2, size=1)
                                neg_i = np.random.choice(
                                    np.setdiff1d(
                                        np.arange(0, len(node_list)), [index1, index2]
                                    ),
                                    size=1,
                                )[0]
                                neg_index1 = index1 if neg_v == 1 else neg_i
                                neg_index2 = index2 if neg_v == 0 else neg_i
                                neg_same_type = (
                                    info.loc[node_list[neg_index1], "highway_enc"]
                                    == info.loc[node_list[neg_index2], "highway_enc"]
                                )
                                train_pairs.append(
                                    (
                                        int(neg_index1),
                                        int(neg_index2),
                                        0,
                                        int(neg_same_type),
                                    )
                                )
        return train_pairs

    def generate_shortest_paths(self, G: nx.Graph, n_shortest_paths: int):
        adj = nx.adjacency_matrix(G)

        def get_path(node_list, Pr, i, j):
            path = [node_list[j]]
            k = j
            while Pr[i, k] != -9999:
                path.append(tuple(node_list[Pr[i, k]]))
                k = Pr[i, k]
            if Pr[i, k] == -9999 and k != i:
                return []

            return path[::-1]

        _, P = shortest_path(
            adj, directed=True, method="D", return_predecessors=True, unweighted=True
        )

        nodes = np.arange(len(G.nodes))
        paths = []
        for source in tqdm(nodes):
            targets = np.random.choice(
                np.setdiff1d(np.arange(len(G.nodes)), [source]), size=n_shortest_paths
            )
            node_paths = []
            for target in targets:
                node_paths.append(get_path(list(G.nodes), P, source, target))
            paths.append(node_paths)

        return paths

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


class SRN2Vec(nn.Module):
    def __init__(self, network, emb_dim):
        super(SRN2Vec, self).__init__()
        self.lin_vx = nn.Linear(len(network.line_graph.nodes), emb_dim)
        self.lin_vy = nn.Linear(len(network.line_graph.nodes), emb_dim)

        self.lin_out = nn.Linear(emb_dim, 3)
        self.act_out = nn.Sigmoid()

    def forward(self, vx, vy):
        x = self.lin_vx(vx)
        y = self.lin_vy(vy)

        x = x * y  # aggregate embeddings

        x = self.lin_out(x)

        return self.act_out(x)
