from operator import itemgetter
from typing import Dict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn import model_selection, neighbors
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .task import Task


class NextLocationPrediciton(Task):
    """
    Next Location Prediction Task. Extract Embeddings for trajectory and put them as sequence into lstm model.
    Predict Next location given partial trajectory without last road segment.
    """

    def __init__(
        self,
        traj_dataset,
        network,
        device,
        seed,
        emb_dim: int = 128,
        batch_size: int = 128,
        epochs: int = 10,
    ):
        self.metrics = {}
        self.data = traj_dataset
        self.network = network
        self.emb_dim = emb_dim
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        # make a train test split on trajectorie data
        train, test = model_selection.train_test_split(
            self.data, test_size=0.2, random_state=self.seed
        )
        self.train_loader = DataLoader(
            NL_Dataset(train, network),
            collate_fn=NL_Dataset.collate_fn_padd,
            batch_size=batch_size,
            shuffle=True,
        )
        self.eval_loader = DataLoader(
            NL_Dataset(test, network),
            collate_fn=NL_Dataset.collate_fn_padd,
            batch_size=batch_size,
        )

    def evaluate(self, emb: np.ndarray, coord_sys: str = "EPSG:3763"):  # porto coord
        model = NL_LSTM(
            out_dim=len(
                self.network.line_graph.nodes
            ),  # predict the destination node (classification)
            device=self.device,
            # pos_map=dict(
            #     zip(
            #         self.network.gdf_edges.fid,
            #         self.network.gdf_edges.geometry.to_crs(
            #             coord_sys
            #         ).centroid,  # portugal specific
            #     )
            # ),
            emb_dim=emb.shape[1],
            batch_size=self.batch_size,
        )

        # train on x trajectories
        model.train_model(loader=self.train_loader, emb=emb, epochs=self.epochs)

        # eval on test set using distance loss
        yh, y = model.predict(loader=self.eval_loader, emb=emb)
        res = {}
        for name, (metric, args) in self.metrics.items():
            res[name] = metric(y, yh, **args)

        return res

    def register_metric(self, name, metric_func, args):
        self.metrics[name] = (metric_func, args)

    # @staticmethod
    # def distance_loss(yh: torch.Tensor, y: torch.Tensor, position_map: Dict, **args):
    #     pred_pos = itemgetter(*yh.tolist())(position_map)
    #     label_pos = itemgetter(*y.tolist())(position_map)
    #     # print(pred_pos[0], label_pos[0], pred_pos[0].distance(label_pos[0]))
    #     loss = 0
    #     for p, l in zip(pred_pos, label_pos):
    #         loss += p.distance(l)
    #     # print(loss)
    #     return torch.tensor(loss, requires_grad=True)


class NL_Dataset(Dataset):
    def __init__(self, data, network):
        self.X = data["seg_seq"].swifter.apply(lambda x: x[:-1]).values
        self.y = data["seg_seq"].swifter.apply(lambda x: x[-1]).values

        self.network = network
        self.map = self.create_edge_emb_mapping()
        self.neighbor_masks = self.create_neighborhood_mask()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=int),
            self.y[idx],
            self.neighbor_masks[idx],
            self.map,
        )

    # tested index mapping is correct
    def create_edge_emb_mapping(self):
        """_summary_
        Maps fid of edges to index of node (i.e edge) in line_graph
        Returns:
            _type_: _description_
        """
        map = {}
        nodes = list(self.network.line_graph.nodes)
        for index, id in zip(self.network.gdf_edges.index, self.network.gdf_edges.fid):
            map[id] = nodes.index(index)
        # print(map == map2) # yields true

        return map

    def create_neighborhood_mask(self):
        """
        Gets the neighbors for each last visible trajectory road segment.
        The neighbors are a list of indices corresponding to the index in
        network.line_graph.nodes of the road segment i.e the index of the embedding.

        Returns:
            list[list[int]]: neighbor idxs as 2D List
        """
        adj = nx.to_numpy_matrix(self.network.line_graph)
        last_items = [x[-1] for x in self.X]  # fid of edges
        # nodes = np.array(self.network.line_graph.nodes, dtype="l,l,l")
        neighbor_list = []

        # get neighbors for each node
        for fid in last_items:
            graph_id = self.map[fid]
            node_neigh = np.flatnonzero(adj[graph_id]).tolist()
            neighbor_list.append(node_neigh)
        # idxs = np.array(
        #     self.network.gdf_edges[
        #         self.network.gdf_edges.fid.isin(last_items)
        #     ].index.to_list(),
        #     dtype="l,l,l",
        # )
        # for i in idxs:
        #     ns = list(self.network.line_graph.neighbors(tuple(i)))
        #     for n in ns:
        #         print(list(self.network.line_graph.nodes).index(n))
        #     print("---")

        return neighbor_list

    @staticmethod
    def collate_fn_padd(batch):
        """
        Padds batch of variable length
        """
        data, label, neigh_masks, map = zip(*batch)
        # seq length for each input in batch
        lengths_old = torch.tensor([t.shape[0] for t in data])

        # sort data for pad packet, since biggest sequence should be first and then descending order
        sort_idxs = torch.argsort(lengths_old, descending=True, dim=0)
        lengths = lengths_old[sort_idxs]
        data = [
            x
            for _, x in sorted(
                zip(lengths_old.tolist(), data), key=lambda pair: pair[0], reverse=True
            )
        ]
        label = [
            x
            for _, x in sorted(
                zip(lengths_old.tolist(), label), key=lambda pair: pair[0], reverse=True
            )
        ]
        neigh_masks = [
            x
            for _, x in sorted(
                zip(lengths_old.tolist(), neigh_masks),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]

        # pad
        data = torch.nn.utils.rnn.pad_sequence(data, padding_value=0, batch_first=True)
        # compute mask
        mask = data != 0
        label = [map[0][l] for l in label]  # map to embedding ids

        return (
            data,
            torch.tensor(label, dtype=torch.long),
            neigh_masks,
            lengths,
            mask,
            map[0],
        )


class NL_LSTM(nn.Module):
    def __init__(
        self,
        out_dim: int,
        device,
        emb_dim: int = 128,
        hidden_units: int = 256,
        layers: int = 1,
        batch_size: int = 128,
    ):
        super(NL_LSTM, self).__init__()
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_units,
            num_layers=layers,
            batch_first=True,  # dropout=0.5
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 2),
            nn.ReLU(),
            nn.Linear(hidden_units * 2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_dim),
        )
        self.soft = nn.Softmax(dim=1)
        self.hidden_units = hidden_units
        self.layers = layers
        self.batch_size = batch_size
        self.device = device
        self.loss = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

        self.encoder.to(device)
        self.decoder.to(device)

    def masked_out(self, x, idxs):
        mask = torch.ones_like(x)
        for row, idx in zip(mask, idxs):
            row[idx] = 0
        x = x + (mask + 1e-44).log()
        # divider = torch.sum(torch.exp(x) * mask, axis=1).reshape(-1, 1)

        return x  # torch.log(torch.div(torch.exp(x) * mask, divider))

    def forward(self, x, lengths, neigh_masks):
        batch_size, seq_len, _ = x.size()
        self.hidden = self.init_hidden(batch_size=batch_size)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        x, _ = self.encoder(x)

        x, plengths = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, padding_value=0
        )
        x = x.contiguous()  # batch x seq x hidden
        # x = x.view(-1, x.shape[2])
        x = torch.stack(
            [x[b, plengths[b] - 1] for b in range(batch_size)]
        )  # get last valid item per batch batch x hidden

        yh = self.decoder(x)

        yh = self.masked_out(yh, neigh_masks)

        return yh  # (batch x len(possible nodes))

    def train_model(self, loader, emb, epochs=100):
        self.train()
        for e in range(epochs):
            total_loss = 0
            for X, y, neigh_masks, lengths, mask, map in loader:
                emb_batch = self.get_embedding(emb, X.clone(), mask, map)
                emb_batch = emb_batch.to(self.device)
                y = y.to(self.device)
                yh = self.forward(emb_batch, lengths, neigh_masks)

                loss = self.loss(yh.squeeze(), y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()

            print(f"Average training loss in episode {e}: {total_loss/len(loader)}")

    def predict(self, loader, emb):
        self.eval()
        yhs, ys = [], []
        for X, y, neigh_mask, lengths, mask, map in loader:
            emb_batch = self.get_embedding(emb, X.clone(), mask, map)
            emb_batch = emb_batch.to(self.device)
            y = y.to(self.device)
            yh = self.soft(self.forward(emb_batch, lengths, neigh_mask)).argmax(dim=1)
            yhs.extend(yh.tolist())
            ys.extend(y.tolist())

        return np.array(yhs), np.array(ys)

    def init_hidden(self, batch_size):
        hidden_a = torch.randn(self.layers, batch_size, self.hidden_units)
        hidden_b = torch.randn(self.layers, batch_size, self.hidden_units)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        hidden_a = hidden_a.to(self.device)
        hidden_b = hidden_b.to(self.device)

        return (hidden_a, hidden_b)

    def get_embedding(self, emb, batch, mask, map):
        """
        Transform batch_size, seq_length, 1 to batch_size, seq_length, emb_size
        """
        res = torch.zeros((batch.shape[0], batch.shape[1], emb.shape[1]))
        for i, seq in enumerate(batch):
            emb_ids = itemgetter(*seq[mask[i]].tolist())(map)
            res[i, mask[i], :] = torch.Tensor(emb[emb_ids, :]).float()

        return res
