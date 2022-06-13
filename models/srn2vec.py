import gc
import json
import os
from itertools import chain, combinations
from platform import node

import networkx as nx
import numpy as np
import pandas as pd
import tables
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import Model


class SRN2VecModel(Model):
    def __init__(
        self,
        data,
        device,
        network,
        emb_dim: int = 128,
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
        self.model = SRN2Vec(network, emb_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_func = nn.BCELoss()
        self.network = network

        # self.loader = DataLoader()

    def generate_data(
        self,
        n_shortest_paths: int = 1280,
        window_size: int = 900,  # 900 meters
        number_negative: int = 7,
        save_batch_size=128,
        file_path=".",
    ):
        save_path = os.path.join(file_path, "srn2vec-traindata.json")
        paths = self.generate_shortest_paths(
            self.network.line_graph, n_shortest_paths=n_shortest_paths
        )
        for i in tqdm(
            range(0, len(paths), save_batch_size * len(self.network.line_graph.nodes))
        ):
            trainset = self.generate_train_pairs(
                paths[i : i + (save_batch_size * len(self.network.line_graph.nodes))],
                window_size,
                number_negative,
            )
            if not os.path.isfile(save_path):
                with open(save_path, "w") as fp:
                    json.dump(trainset, fp)
            else:
                with open(save_path, "r") as fp:
                    a = np.array(json.load(fp))
                    a = np.unique(np.vstack([a, np.array(trainset)]), axis=0)
                with open(save_path, "w") as fp:
                    json.dump(a.tolist(), fp)

    def generate_train_pairs(self, paths: list, window_size: int, number_negative: int):
        # generate windows and extract features from paper (same inter type and same degree)
        train_pairs = []
        info = pd.DataFrame(self.network.gdf_edges)
        node_list = np.array(self.network.line_graph.nodes, dtype="l,l,l")
        node_idx_to_length_map = info.loc[node_list.astype(list), "length"].to_numpy()
        node_idx_to_highway_map = info.loc[node_list.astype(list), "highway"].to_numpy()

        pairs = self.extract_pairs(
            node_idx_to_length_map,
            node_idx_to_highway_map,
            node_list,
            paths,
            window_size,
            number_negative,
        )

        return pairs

    def extract_pairs(
        self,
        info_length,
        info_highway,
        node_list,
        node_paths,
        window_size,
        number_negative,
    ):
        res = []
        orig_lengths = np.array(
            [0] + [len(x) for x in node_paths]
        ).cumsum()  # lengths of orginal sequences in flatted
        flatted = list(chain.from_iterable(node_paths))
        # flat array and save indices -> loc on all and reshape after
        # get all lengths of sequence roads
        flat_lengths = info_length[flatted]

        node_combs = []
        for i in range(len(orig_lengths) - 1):
            lengths = flat_lengths[orig_lengths[i] : orig_lengths[i + 1]]
            # cumsum = lengths.cumsum()
            for j in range(len(lengths)):
                mask = (lengths[j:].cumsum() < window_size).sum()
                # idx = (np.abs(lengths[j:].cumsum() - window_size)).argmin()
                window = node_paths[i][j : j + mask]
                if len(window) > 1:
                    combs = tuple(combinations(window, 2))
                    node_combs.extend(combs)

        # save distinct tuples
        node_combs = list(dict.fromkeys(node_combs))
        node_combs = list(chain.from_iterable(node_combs))

        highways = info_highway[node_combs].reshape(int(len(node_combs) / 2), 2)
        pairs = np.c_[
            np.array(node_combs).reshape(int(len(node_combs) / 2), 2),
            np.ones(highways.shape[0]),
            highways[:, 0] == highways[:, 1],
        ].astype(
            int
        )  # same type

        res.extend(tuple(pairs.tolist()))

        # generate negative sample
        neg_nodes = np.random.choice(
            np.setdiff1d(np.arange(0, len(node_list)), node_combs),
            size=pairs.shape[0] * number_negative,
        )

        neg_pairs = pairs.copy()
        neg_pairs = neg_pairs.repeat(repeats=number_negative, axis=0)
        replace_mask = np.random.randint(0, 2, size=neg_pairs.shape[0]).astype(bool)
        neg_pairs[replace_mask, 0] = neg_nodes[replace_mask]
        neg_pairs[~replace_mask, 1] = neg_nodes[~replace_mask]
        neg_pairs[:, 2] -= 1

        neg_highways = info_highway[neg_pairs[:, :2].flatten()].reshape(
            neg_pairs.shape[0], 2
        )

        neg_pairs[:, 3] = neg_highways[:, 0] == neg_highways[:, 1]

        res.extend(tuple(neg_pairs.tolist()))

        return res

    def generate_shortest_paths(self, G: nx.Graph, n_shortest_paths: int):
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
        for source in tqdm(nodes):
            targets = np.random.choice(
                np.setdiff1d(np.arange(len(G.nodes)), [source]),
                size=n_shortest_paths,
                replace=False,
            )
            node_paths = []
            for target in targets:
                path = get_path(P, source, target)
                if path != []:
                    node_paths.append(path)
            paths.extend(tuple(node_paths))

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


class SRN2Vec_Dataset(Dataset):
    def __init__(self, data):
        self.network = network
        self.map = self.create_edge_emb_mapping()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=int), self.y[idx], self.map
