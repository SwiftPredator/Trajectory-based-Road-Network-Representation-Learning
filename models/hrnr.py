import os
import pickle
from turtle import forward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import add_self_loops, negative_sampling, remove_self_loops

from .hrnr_original.gcn_layers import *
from .model import Model
from .training.hrnr_data_generation import create_adj, create_features


class HRNRModel(Model):
    """Adapter class for hrnr model"""

    def __init__(
        self,
        data,
        device,
        network,
        data_path=None,
        emb_dim=128,
        remove_highway_label=False,
    ):
        self.network = network
        self.device = device
        self.data = data
        self.remove_highway_label = remove_highway_label
        self.get_data(path=data_path)

        self.model = GraphEncoderTL(
            self.struct_assign, self.fnc_assign, self.struct_adj, self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model = self.model.to(self.device)

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model(
                self.node_feature,
                self.type_feature,
                self.length_feature,
                self.lane_feature,
                self.adj,
            )
            loss = self.recon_loss(z, self.data.edge_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 50 == 0:
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

    def get_data(self, path: str):
        struct_assign = pickle.load(
            open(os.path.join(path, "struct_assign_porto"), "rb")
        )
        fnc_assign = pickle.load(open(os.path.join(path, "fnc_assign_porto"), "rb"))

        struct_assign = torch.tensor(
            struct_assign, dtype=torch.float, device=self.device
        )
        fnc_assign = torch.tensor(fnc_assign, dtype=torch.float, device=self.device)

        node_count_interest = len(self.network.line_graph.nodes)
        adj = create_adj(self.network)
        features = create_features(
            self.network, remove_highway_label=self.remove_highway_label
        )
        adj = adj[:node_count_interest, :node_count_interest]
        features = features[:node_count_interest]
        struct_assign = struct_assign[:node_count_interest]

        # features = features.tolist()
        # while len(features) < 16000:
        #     features.append([0, 0, 0, 0])
        # features = np.array(features)

        self_loop = np.eye(len(adj))
        adj = np.logical_or(adj, self_loop)
        adj = sparse.coo_matrix(adj, dtype=float)

        adj_indices = torch.tensor(
            np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1),
            dtype=torch.long,
        ).t()
        adj_values = torch.tensor(adj.data, dtype=torch.float)
        adj_shape = adj.shape
        adj = torch.sparse_coo_tensor(
            adj_indices, adj_values, adj_shape, device=self.device
        )

        special_spmm = SpecialSpmm()
        edge = adj._indices().to(self.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.device)
        struct_inter = special_spmm(
            edge, edge_e, torch.Size([adj.shape[0], adj.shape[1]]), struct_assign
        )  # N*N   N*C
        struct_adj = torch.mm(struct_assign.t(), struct_inter)

        self.adj = adj
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign
        self.struct_adj = struct_adj
        self.lane_feature = torch.tensor(
            features[:, 0], dtype=torch.long, device=self.device
        )
        self.type_feature = torch.tensor(
            features[:, 1], dtype=torch.long, device=self.device
        )
        self.length_feature = torch.tensor(
            features[:, 2], dtype=torch.long, device=self.device
        )
        self.node_feature = torch.tensor(
            features[:, 3], dtype=torch.long, device=self.device
        )

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
                self.node_feature,
                self.type_feature,
                self.length_feature,
                self.lane_feature,
                self.adj,
            )
            .detach()
            .cpu()
            .numpy()
        )


class GraphEncoderTL(nn.Module):
    def __init__(self, struct_assign, fnc_assign, struct_adj, device):
        super(GraphEncoderTL, self).__init__()
        # hyperparameters from original paper
        node_num = 16000
        node_dims = 32
        type_num = 20
        type_dims = 32
        length_num = 2200
        length_dims = 32
        lane_num = 6
        lane_dims = 32
        self.gnn_layers = 1

        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign
        self.struct_adj = struct_adj

        self.node_emb_layer = nn.Embedding(node_num, node_dims).to(device)
        self.type_emb_layer = nn.Embedding(type_num, type_dims).to(device)
        self.length_emb_layer = nn.Embedding(length_num, length_dims).to(device)
        self.lane_emb_layer = nn.Embedding(lane_num, lane_dims).to(device)

        self.tl_layer = GraphEncoderTLCore(
            self.struct_assign, self.fnc_assign, device=device
        )

        self.init_feat = None

    def forward(self, node_feature, type_feature, length_feature, lane_feature, adj):
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        self.init_feat = raw_feat
        # print(self.struct_adj.shape, raw_feat.shape, adj.shape)
        for _ in range(self.gnn_layers):
            raw_feat = self.tl_layer(self.struct_adj, raw_feat, adj)

        return raw_feat


class GraphEncoderTLCore(Module):
    def __init__(self, struct_assign, fnc_assign, device):
        super(GraphEncoderTLCore, self).__init__()
        # hyperparameters from original paper for label prediciton task
        hidden_dims = 128

        self.raw_struct_assign = struct_assign
        self.raw_fnc_assign = fnc_assign

        self.fnc_gcn = GraphConvolution(
            in_features=hidden_dims,
            out_features=hidden_dims,
            device=device,
        ).to(device)

        self.struct_gcn = GraphConvolution(
            in_features=hidden_dims,
            out_features=hidden_dims,
            device=device,
        ).to(device)

        self.node_gat = SPGCN(
            in_features=hidden_dims,
            out_features=hidden_dims,
            device=device,
        ).to(device)

        self.l_c = torch.nn.Linear(hidden_dims * 2, 1).to(device)

        self.l_s = torch.nn.Linear(hidden_dims * 2, 1).to(device)

        self.l_i = torch.nn.Linear(hidden_dims, hidden_dims).to(device)

        self.sigmoid = nn.Sigmoid()

        self.batch_norm_1 = nn.BatchNorm1d(hidden_dims, eps=1e-12).cuda()
        self.batch_norm_2 = nn.BatchNorm1d(hidden_dims, eps=1e-12).cuda()

        self.device = device

    def forward(self, struct_adj, raw_feat, raw_adj):
        self.struct_assign = self.raw_struct_assign / (
            F.relu(torch.sum(self.raw_struct_assign, 0) - 1.0) + 1.0
        )
        self.fnc_assign = self.raw_fnc_assign / (
            F.relu(torch.sum(self.raw_fnc_assign, 0) - 1.0) + 1.0
        )
        self.struct_emb = torch.mm(self.struct_assign.t(), raw_feat)
        self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)

        self.fnc_emb_adj = self.batch_norm_1(self.fnc_emb)
        self.fnc_adj = F.sigmoid(
            torch.mm(self.fnc_emb_adj, self.fnc_emb_adj.t())
        )  # n_f * n_f
        self.fnc_adj = self.fnc_adj

        self.fnc_emb = self.fnc_gcn(
            self.fnc_emb.unsqueeze(0), self.fnc_adj.unsqueeze(0)
        ).squeeze()
        fnc_message = torch.div(
            torch.mm(self.fnc_assign, self.fnc_emb),
            (F.relu(torch.sum(self.fnc_assign, 1) - 1.0) + 1.0).unsqueeze(1) * 50.0,
        )

        self.r_f = self.sigmoid(self.l_c(torch.cat((self.struct_emb, fnc_message), 1)))
        self.struct_emb = self.struct_emb + self.r_f * 0.2 * fnc_message
        struct_adj = (
            F.relu(
                struct_adj - torch.eye(struct_adj.shape[1]).to(self.device) * 10000.0
            )
            + torch.eye(struct_adj.shape[1]).to(self.device) * 1.0
        )
        self.struct_emb = self.struct_gcn(
            self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)
        ).squeeze()

        struct_message = torch.mm(self.raw_struct_assign, self.struct_emb)
        self.r_s = self.sigmoid(self.l_s(torch.cat((raw_feat, struct_message), 1)))
        raw_feat = self.node_gat(raw_feat, raw_adj)
        #    raw_feat = raw_feat + self.r_s * 0.7 * struct_message
        return raw_feat
