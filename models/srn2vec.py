import json
import os
from itertools import chain, combinations

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import Model


class SRN2VecModel(Model):
    def __init__(
        self,
        data,  # placeholder
        device,
        network,
        emb_dim: int = 128,
    ):
        """
        Initialize SRN2Vec
        Args:
            data (_type_): placeholder
            device (_type_): torch device
            network (nx.Graph): graph of city where nodes are intersections and edges are roads
            emb_dim (int, optional): embedding dimension. Defaults to 128.
        """
        self.device = device
        self.emb_dim = emb_dim
        self.model = SRN2Vec(network, device=device, emb_dim=emb_dim, out_dim=2)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = nn.BCELoss()
        self.network = network
        # self.data = data

    def train(self, epochs: int = 1000, batch_size: int = 128):
        """
        Train the SRN2Vec Model (load dataset before with .load_data())
        Args:
            epochs (int, optional): epochs to train. Defaults to 1000.
            batch_size (int, optional): batch_size. Defaults to 128.
        """
        self.model.to(self.device)
        loader = DataLoader(
            SRN2VecDataset(self.data, len(self.network.line_graph.nodes)),
            batch_size=batch_size,
            shuffle=True,
        )
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for i, (X, y) in enumerate(loader):
                X = X.to(self.device)
                y = y.to(self.device)

                self.optim.zero_grad()
                yh = self.model(X)
                loss = self.loss_func(yh.squeeze(), y)

                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                if i % 1000 == 0:
                    print(
                        f"Epoch: {e}, Iteration: {i}, sample_loss: {loss.item()}, Avg. Loss: {total_loss/(i+1)}"
                    )

            print(f"Average training loss in episode {e}: {total_loss/len(loader)}")

    def generate_data(
        self,
        n_shortest_paths: int = 1280,
        window_size: int = 900,  # 900 meters
        number_negative: int = 7,
        save_batch_size=128,
        file_path=".",
    ):
        """
        Generates the dataset like described in the corresponding paper. Since this needs alot of ram we use a batching approach.
        Args:
            n_shortest_paths (int, optional): how many shortest paths per node in graph. Defaults to 1280.
            window_size (int, optional): window size for distance neighborhood in meters. Defaults to 900.
            number_negative (int, optional): Negative samples to draw per positive sample.
            save_batch_size (int, optional): how many shortest paths to process between each save. Defaults to 128.
            file_path (str, optional): path where the dataset should be saved. Defaults to ".".
        """
        save_path = os.path.join(file_path, "srn2vec-traindata.json")
        paths = self.generate_shortest_paths(
            self.network.line_graph, n_shortest_paths=n_shortest_paths
        )
        # Iterate over batches
        for i in tqdm(
            range(0, len(paths), save_batch_size * len(self.network.line_graph.nodes))
        ):
            trainset = self.generate_train_pairs(
                paths[i : i + (save_batch_size * len(self.network.line_graph.nodes))],
                window_size,
                number_negative,
            )
            # save batch
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
        """
        Generates the traning pairs consisting of (v_x, v_y, in window_size?, same road type?). This is highly optimized.
        Args:
            paths (list): shortest path to process as 2D list
            window_size (int): window_size in meters
            number_negative (int): number of negative samples to draw for each positive sample

        Returns:
            list: training pairs
        """
        # extraxt static data beforehand to improve speed
        info = pd.DataFrame(self.network.gdf_edges)
        node_list = np.array(self.network.line_graph.nodes, dtype="l,l,l")
        node_idx_to_length_map = info.loc[node_list.astype(list), "length"].to_numpy()
        node_idx_to_highway_map = info.loc[node_list.astype(list), "highway"].to_numpy()

        # generate pairs
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
        info_length: np.array,
        info_highway: np.array,
        node_list: np.array,
        node_paths: list,
        window_size: int,
        number_negative: int,
    ):
        """_summary_

        Args:
            info_length (np.array): length for each node in graph (ordered by node ordering in graph)
            info_highway (np.array): type for each node in graph (ordered by node ordering in graph)
            node_list (np.array): nodes in graph (ordered by node ordering in graph)
            node_paths (list): shortest paths
            window_size (int): window_size in meters
            number_negative (int): number negative to draw for each node

        Returns:
            list: training pairs
        """
        res = []
        # lengths of orginal sequences in flatted with cumsum to get real position in flatted
        orig_lengths = np.array([0] + [len(x) for x in node_paths]).cumsum()
        flatted = list(chain.from_iterable(node_paths))
        # get all lengths of sequence roads
        flat_lengths = info_length[flatted]

        # generate window tuples
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

        # generate same type label
        highways = info_highway[node_combs].reshape(int(len(node_combs) / 2), 2)
        pairs = np.c_[
            np.array(node_combs).reshape(int(len(node_combs) / 2), 2),
            np.ones(highways.shape[0]),
            highways[:, 0] == highways[:, 1],
        ].astype(
            int
        )  # same type

        res.extend(tuple(pairs.tolist()))

        # generate negative sample with same procedure as for positive
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
        """
        Generates shortest paths between node in graph G.
        Args:
            G (nx.Graph): graph
            n_shortest_paths (int): how many shortest path to generate per node in G.

        Returns:
            list: shortest_paths
        """
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

    def load_dataset(self, path: str):
        with open(path, "r") as fp:
            self.data = np.array(json.load(fp))

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), os.path.join(path + "/model.pt"))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        ...

    def load_emb(self):
        return self.model.embedding.weight.data


class SRN2Vec(nn.Module):
    def __init__(self, network, device, emb_dim: int = 128, out_dim: int = 2):
        super(SRN2Vec, self).__init__()
        self.embedding = nn.Embedding(len(network.line_graph.nodes), emb_dim)
        self.lin_vx = nn.Linear(emb_dim, emb_dim)
        self.lin_vy = nn.Linear(emb_dim, emb_dim)

        self.lin_out = nn.Linear(emb_dim, out_dim)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        # y_emb = self.embedding(vy)

        # x = self.lin_vx(emb[:, 0])
        # y = self.lin_vy(emb[:, 1])
        x = emb[:, 0, :] * emb[:, 1, :]  # aggregate embeddings

        x = self.lin_out(x)

        yh = self.act_out(x)

        return yh


class SRN2VecDataset(Dataset):
    def __init__(self, data, num_classes: int):
        self.X = data[:, :2]
        self.y = data[:, 2:]
        self.num_cls = num_classes

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=int), torch.Tensor(
            self.y[idx]
        )  # F.one_ont(self.X[idx], self.num_cls)
