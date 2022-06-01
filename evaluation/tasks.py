from abc import ABC, abstractclassmethod
from operator import itemgetter
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn import model_selection
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Task(ABC):
    """
    Base class for all evaluation tasks
    """

    @abstractclassmethod
    def evaluate(emb: np.ndarray):
        ...


class RoadTypeClfTask(Task):
    def __init__(self, decoder, y):
        self.decoder = decoder
        self.metrics = {}
        self.y = y
        # Use same split for every evaluation
        (
            self.x_train_idxs,
            self.x_test_idxs,
            self.y_train_idxs,
            self.y_test_idxs,
        ) = model_selection.train_test_split(
            np.arange(y.shape[0]), np.arange(y.shape[0]), test_size=0.2, random_state=1
        )

    def evaluate(self, emb: np.ndarray) -> Dict[str, any]:
        # create train/test split @TODO: Move in init since each embed is evaluated on other split -> Done
        X_train, X_test, y_train, y_test = (
            emb[self.x_train_idxs],
            emb[self.x_test_idxs],
            self.y[self.y_train_idxs],
            self.y[self.y_test_idxs],
        )

        # train decoder
        decoder = clone(self.decoder)
        decoder.fit(X_train, y_train)

        # calculate metrics
        res = {}
        for name, (metric, args, proba) in self.metrics.items():
            if proba:
                res[name] = metric(y_test, decoder.predict_proba(X_test), **args)
            else:
                res[name] = metric(y_test, decoder.predict(X_test), **args)

        return res

    def register_metric(self, name, metric_func, args, proba=False):
        self.metrics[name] = (metric_func, args, proba)


class TravelTimeEstimation(Task):
    """
    Travel Time Estimation Task. Extract Embeddings for trajectory and put them as sequence into lstm model.
    Predict traverse time.
    """

    def __init__(
        self, traj_dataset, network, device, emb_dim: int = 128, batch_size: int = 128
    ):
        self.metrics = {}
        self.data = traj_dataset
        self.network = network
        self.emb_dim = emb_dim
        self.device = device
        self.batch_size = batch_size
        # make a train test split on trajectorie data
        train, test = model_selection.train_test_split(
            self.data, test_size=0.2, random_state=1
        )
        self.train_loader = DataLoader(
            TTE_Dataset(train, network),
            collate_fn=TTE_Dataset.collate_fn_padd,
            batch_size=batch_size,
            shuffle=True,
        )
        self.eval_loader = DataLoader(
            TTE_Dataset(test, network),
            collate_fn=TTE_Dataset.collate_fn_padd,
            batch_size=batch_size,
        )

    def evaluate(self, emb: np.ndarray):
        model = TTE_LSTM(
            device=self.device, emb_dim=self.emb_dim, batch_size=self.batch_size
        )

        # train on x trajectories
        model.train_model(loader=self.train_loader, emb=emb)

        # eval on rest
        yh, y = model.predict(loader=self.eval_loader, emb=emb)
        res = {}
        for name, (metric, args) in self.metrics.items():
            res[name] = metric(y, yh, **args)

        return res

    def register_metric(self, name, metric_func, args):
        self.metrics[name] = (metric_func, args)


class TTE_Dataset(Dataset):
    def __init__(self, data, network):
        self.X = data["seg_seq"].values
        self.y = data["travel_time"].values
        self.network = network
        self.map = self.create_edge_emb_mapping()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=int), self.y[idx], self.map

    def create_edge_emb_mapping(self):
        # map from trajectory edge id to embedding id
        edge_ids = np.array(self.network.gdf_edges.index, dtype="i,i,i")
        traj_edge_idx = np.array(self.network.gdf_edges.fid)
        node_ids = np.array(self.network.line_graph.nodes, dtype="i,i,i")
        sort_idx = node_ids.argsort()
        emb_ids = sort_idx[np.searchsorted(node_ids, edge_ids, sorter=sort_idx)]

        map = dict(zip(traj_edge_idx, emb_ids))

        return map

    @staticmethod
    def collate_fn_padd(batch):
        """
        Padds batch of variable length
        """
        data, label, map = zip(*batch)
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

        # pad
        data = torch.nn.utils.rnn.pad_sequence(data, padding_value=-1, batch_first=True)
        # compute mask
        mask = data != -1

        return data, torch.Tensor(label), lengths, mask, map[0]


class TTE_LSTM(nn.Module):
    def __init__(
        self,
        device,
        emb_dim: int = 128,
        hidden_units: int = 256,
        layers: int = 1,
        batch_size: int = 128,
    ):
        super(TTE_LSTM, self).__init__()
        self.encoder = nn.LSTM(emb_dim, hidden_units, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Linear(hidden_units, 1)
        )
        self.hidden_units = hidden_units
        self.layers = layers
        self.batch_size = batch_size
        self.device = device
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.encoder.to(device)
        self.decoder.to(device)

    def forward(self, x, lengths):
        batch_size, seq_len, _ = x.size()
        self.hidden = self.init_hidden(batch_size=batch_size)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        x, self.hidden = self.encoder(x, self.hidden)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, padding_value=-1
        )
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        yh = self.decoder(x)

        yh = yh.view(batch_size, seq_len, -1)

        return yh  # (batch x seq x 1)

    def train_model(self, loader, emb, epochs=100):
        self.train()
        for e in range(epochs):
            total_loss = 0
            for X, y, lengths, mask, map in tqdm(loader):
                emb_batch = self.get_embedding(emb, X.clone(), mask, map)
                emb_batch = emb_batch.to(self.device)
                y = y.to(self.device)
                yh = self.forward(emb_batch, lengths).view(-1, 1)
                label_pos = (lengths - 1) + torch.arange(
                    0, emb_batch.shape[0]
                ) * lengths[0]
                yh = yh[label_pos]
                # print(yh.shape, y.shape)
                loss = self.loss(yh.squeeze(), y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()

            # print(f"Average training loss in episode {e}: {total_loss/len(loader)}")

    def predict(self, loader, emb):
        self.eval()
        yhs, ys = [], []
        for X, y, lengths, mask, map in tqdm(loader):
            emb_batch = self.get_embedding(emb, X.clone(), mask, map)
            emb_batch = emb_batch.to(self.device)
            y = y.to(self.device)
            yh = self.forward(emb_batch, lengths).view(-1, 1)
            label_pos = (lengths - 1) + torch.arange(0, emb_batch.shape[0]) * lengths[0]
            yh = yh[label_pos]
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
        res = batch.unsqueeze(-1).repeat(1, 1, emb.shape[1]).float()
        for i, seq in enumerate(batch):
            # transform sequence
            # map from trajectory id to embedding id
            # edge_ids = np.array(
            #     network.gdf_edges.iloc[seq[mask[i]]].index, dtype="i,i,i"
            # )
            # node_ids = np.array(network.line_graph.nodes, dtype="i,i,i")
            # # emb_ids = [np.where(eid == node_ids)[0][0] for eid in edge_ids]
            # # print(seq, edge_ids, emb_ids)
            # sort_idx = node_ids.argsort()
            emb_ids = itemgetter(*seq[mask[i]].tolist())(map)
            res[i, mask[i], :] = torch.Tensor(emb[emb_ids, :]).float()

        return res
