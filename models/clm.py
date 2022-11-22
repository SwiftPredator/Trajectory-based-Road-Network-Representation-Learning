import math
import os
from datetime import datetime
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_adj
from tqdm import tqdm

from .model import Model
from .utils import generate_trajid_to_nodeid


def jsd(z1, z2, pos_mask):
    neg_mask = 1 - pos_mask

    sim_mat = torch.mm(z1, z2.t())
    E_pos = math.log(2.0) - F.softplus(-sim_mat)
    E_neg = F.softplus(-sim_mat) + sim_mat - math.log(2.0)
    return (E_neg * neg_mask).sum() / neg_mask.sum() - (
        E_pos * pos_mask
    ).sum() / pos_mask.sum()


def nce(z1, z2, pos_mask):
    sim_mat = torch.mm(z1, z2.t())
    return nn.BCEWithLogitsLoss(reduction="none")(sim_mat, pos_mask).sum(1).mean()


def ntx(z1, z2, pos_mask, tau=0.5, normalize=False):
    if normalize:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    sim_mat = torch.mm(z1, z2.t())
    sim_mat = torch.exp(sim_mat / tau)
    return -torch.log(
        (sim_mat * pos_mask).sum(1) / sim_mat.sum(1) / pos_mask.sum(1)
    ).mean()


def node_node_loss(node_rep1, node_rep2, measure):
    num_nodes = node_rep1.shape[0]

    pos_mask = torch.eye(num_nodes).cuda()

    if measure == "jsd":
        return jsd(node_rep1, node_rep2, pos_mask)
    elif measure == "nce":
        return nce(node_rep1, node_rep2, pos_mask)
    elif measure == "ntx":
        return ntx(node_rep1, node_rep2, pos_mask)


def seq_seq_loss(seq_rep1, seq_rep2, measure):
    batch_size = seq_rep1.shape[0]

    pos_mask = torch.eye(batch_size).cuda()

    if measure == "jsd":
        return jsd(seq_rep1, seq_rep2, pos_mask)
    elif measure == "nce":
        return nce(seq_rep1, seq_rep2, pos_mask)
    elif measure == "ntx":
        return ntx(seq_rep1, seq_rep2, pos_mask)


def node_seq_loss(node_rep, seq_rep, sequences, measure):
    batch_size = seq_rep.shape[0]
    num_nodes = node_rep.shape[0]

    pos_mask = torch.zeros((batch_size, num_nodes + 1)).cuda()
    for row_idx, row in enumerate(sequences):
        pos_mask[row_idx][row] = 1.0
    pos_mask = pos_mask[:, :-1]

    if measure == "jsd":
        return jsd(seq_rep, node_rep, pos_mask)
    elif measure == "nce":
        return nce(seq_rep, node_rep, pos_mask)
    elif measure == "ntx":
        return ntx(seq_rep, node_rep, pos_mask)


def weighted_ns_loss(node_rep, seq_rep, weights, measure):
    if measure == "jsd":
        return jsd(seq_rep, node_rep, weights)
    elif measure == "nce":
        return nce(seq_rep, node_rep, weights)
    elif measure == "ntx":
        return ntx(seq_rep, node_rep, weights)


def random_mask(x, mask_token, mask_prob=0.2):
    mask_pos = (
        torch.empty(x.size(), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < mask_prob
    )
    x = x.clone()
    x[mask_pos] = mask_token
    return


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            input_size, num_heads, hidden_size, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size, encoder_layer, num_layers, activation):
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = [encoder_layer(input_size, output_size)]
        for _ in range(1, num_layers):
            self.layers.append(encoder_layer(output_size, output_size))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.activation(self.layers[i](x, edge_index))
        return x


def train_data_loader(df, padding_id, traj_map):
    min_len = 10
    max_len = 100
    num_samples = 250000

    df["path_len"] = df["path"].map(len)
    df = df.loc[(df["path_len"] > min_len) & (df["path_len"] < max_len)]
    if len(df) > num_samples:
        df = df.sample(n=num_samples, replace=False, random_state=1)

    arr = np.full([num_samples, max_len], padding_id, dtype=np.int32)
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        path_arr = np.array(row["path"], dtype=np.int32)
        traj = list(itemgetter(*path_arr)(traj_map))
        arr[i, : row["path_len"]] = traj

    return torch.LongTensor(arr), None


def next_batch_index(ds, bs, shuffle=True):
    num_batches = math.ceil(ds / bs)

    index = np.arange(ds)
    if shuffle:
        index = np.random.permutation(index)

    for i in range(num_batches):
        if i == num_batches - 1:
            batch_index = index[bs * i :]
        else:
            batch_index = index[bs * i : bs * (i + 1)]
        yield batch_index


class CLMModel(Model):
    def __init__(
        self,
        data,
        device,
        network,
        trans_adj=None,
        traj_data=None,
        emb_dim=128,
        hidden_size=128,
        batch_size=64,
    ):
        graph_encoder1 = GraphEncoder(emb_dim, hidden_size, GATConv, 2, nn.ReLU())
        graph_encoder2 = GraphEncoder(emb_dim, hidden_size, GATConv, 2, nn.ReLU())
        seq_encoder = TransformerModel(hidden_size, 4, hidden_size, 2, 0.2)
        self.model = CLMEncoder(
            vocab_size=data.x.shape[0],
            embed_size=emb_dim,
            hidden_size=hidden_size,
            edge_index1=data.edge_index.cuda(),
            edge_index2=trans_adj,
            graph_encoder1=graph_encoder1,
            graph_encoder2=graph_encoder2,
            seq_encoder=seq_encoder,
        )
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=1e-6
        )

        self.traj_data = traj_data
        self.data = data
        self.num_nodes = data.x.shape[0]
        self.batch_size = batch_size
        self.traj_map = generate_trajid_to_nodeid(network)
        self.device = device

    def train(self, epochs=5):
        l_st = 0.8
        l_ss = l_tt = 0.5 * (1 - l_st)
        for e in range(epochs):
            avg_loss = 0
            data, _ = train_data_loader(self.traj_data, self.num_nodes, self.traj_map)
            for n, batch_index in enumerate(
                next_batch_index(data.shape[0], self.batch_size)
            ):
                data_batch = data[batch_index].cuda()
                self.model.train()
                self.optimizer.zero_grad()
                node_rep1, node_rep2, seq_rep1, seq_rep2 = self.model(data_batch)
                loss_ss = node_node_loss(node_rep1, node_rep2, "jsd")
                loss_tt = seq_seq_loss(seq_rep1, seq_rep2, "jsd")
                loss_st1 = node_seq_loss(node_rep1, seq_rep2, data_batch, "jsd")
                loss_st2 = node_seq_loss(node_rep2, seq_rep1, data_batch, "jsd")
                loss_st = (loss_st1 + loss_st2) / 2
                loss = l_ss * loss_ss + l_tt * loss_tt + l_st * loss_st
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

                if not (n + 1) % 200:
                    t = datetime.now().strftime("%m-%d %H:%M:%S")
                    print(
                        f"{t} | (Train) | Epoch={e}, batch={n + 1} loss={loss.item():.4f}, loss_ss={loss_ss.item():.4f},  loss_tt={loss_tt.item():.4f},  loss_st1={loss_st1.item():.4f}, loss_st2={loss_st2.item():.4f}"
                    )
            print("Epoch: {}, loss: {}".format(e, avg_loss))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def load_emb(self):
        emb, _, _ = self.model.encode_graph()
        return emb.detach().cpu().numpy()


class CLMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        edge_index1,
        edge_index2,
        graph_encoder1,
        graph_encoder2,
        seq_encoder,
    ):
        super(CLMEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.node_embedding = nn.Embedding(vocab_size, embed_size)
        self.padding = torch.zeros(1, hidden_size, requires_grad=False).cuda()
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.graph_encoder1 = graph_encoder1
        self.graph_encoder2 = graph_encoder2
        self.seq_encoder = seq_encoder

    def encode_graph(self):
        node_emb = self.node_embedding.weight
        node_enc1 = self.graph_encoder1(node_emb, self.edge_index1)
        node_enc2 = self.graph_encoder2(node_emb, self.edge_index2)
        return node_enc1 + node_enc2, node_enc1, node_enc2

    def encode_sequence(self, sequences):
        _, node_enc1, node_enc2 = self.encode_graph()

        batch_size, max_seq_len = sequences.size()
        src_key_padding_mask = sequences == self.vocab_size
        pool_mask = (1 - src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1)

        lookup_table1 = torch.cat([node_enc1, self.padding], 0)
        seq_emb1 = (
            torch.index_select(lookup_table1, 0, sequences.view(-1))
            .view(batch_size, max_seq_len, -1)
            .transpose(0, 1)
        )
        seq_enc1 = self.seq_encoder(seq_emb1, None, src_key_padding_mask)
        seq_pooled1 = (seq_enc1 * pool_mask).sum(0) / pool_mask.sum(0)

        lookup_table2 = torch.cat([node_enc2, self.padding], 0)
        seq_emb2 = (
            torch.index_select(lookup_table2, 0, sequences.view(-1))
            .view(batch_size, max_seq_len, -1)
            .transpose(0, 1)
        )
        seq_enc2 = self.seq_encoder(seq_emb2, None, src_key_padding_mask)
        seq_pooled2 = (seq_enc2 * pool_mask).sum(0) / pool_mask.sum(0)
        return seq_pooled1 + seq_pooled2, seq_pooled1, seq_pooled2

    def forward(self, sequences):
        _, node_enc1, node_enc2 = self.encode_graph()
        _, seq_pooled1, seq_pooled2 = self.encode_sequence(sequences)
        return node_enc1, node_enc2, seq_pooled1, seq_pooled2
