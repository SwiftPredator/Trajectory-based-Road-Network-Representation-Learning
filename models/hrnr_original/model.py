# coding=utf-8
import pickle
from platform import node

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from .cmt_gen import get_cmt
from .gcn_layers import *


class ETAModelMLP(Module):
    def __init__(self, hparams):
        super(ETAModelMLP, self).__init__()
        self.hparams = hparams
        self.BiGRU = GRU(self.hparams).to(self.hparams.device)

        self.linear = torch.nn.Linear(hparams.node_dims * 2, 1)

        self.embedding_size = hparams.node_dims
        self.vocab_size = hparams.vocab_size
        self.embedding = Parameter(
            torch.FloatTensor(hparams.node_num, hparams.node_dims)
        )
        torch.nn.init.xavier_uniform(self.embedding)
        self.graph_encoder = GraphEncoder(hparams).to(self.hparams.device)

    def forward(self, token, adj):
        self.graph_embedding = self.graph_encoder(self.embedding, adj)
        start_emb = self.graph_embedding[token[:, 0]]
        end_emb = self.graph_embedding[token[:, -1]]
        cat_emb = torch.cat([start_emb, end_emb], 1)
        pred = self.linear(cat_emb)
        return pred


class ETAModel(Module):
    def __init__(self, hparams):
        super(ETAModel, self).__init__()
        self.hparams = hparams
        self.BiGRU = GRU(self.hparams).to(self.hparams.device)

        self.linear = torch.nn.Linear(hparams.gru_hidden_size * 2, 1)

        self.embedding_size = hparams.node_dims
        self.vocab_size = hparams.vocab_size
        self.embedding = Parameter(
            torch.FloatTensor(hparams.node_num, hparams.node_dims)
        )
        torch.nn.init.xavier_uniform(self.embedding)
        self.graph_encoder = GraphEncoder(hparams).to(self.hparams.device)

    def forward(self, token, adj):
        self.graph_embedding = self.graph_encoder(self.embedding, adj)
        output, _ = self.BiGRU(
            token, self.graph_embedding
        )  # batch_size * max_len * hidden_dims
        output = output.view(
            -1, self.hparams.batch_size, 2, self.hparams.gru_hidden_size
        )
        output = torch.cat([output[:, :, 0, :], output[:, :, 1, :]], 2)
        max_out, _ = torch.max(output, 0)  # batch_size * hidden_dims
        pred = self.linear(max_out)
        return pred


class GRU(nn.Module):
    def __init__(self, hparams):
        super(GRU, self).__init__()
        self.device = hparams.device
        self.emb_size = hparams.node_dims
        self.node_num = hparams.node_num
        self.hidden_size = hparams.gru_hidden_size
        self.batch_size = hparams.batch_size
        self.is_bigru = hparams.is_bigru
        self.state_num = hparams.state_num
        self.gru = nn.GRU(self.emb_size, self.hidden_size, bidirectional=self.is_bigru)
        self.out = nn.Linear(self.hidden_size, self.node_num)

    def forward(self, token, graph_embedding):
        token_emb = graph_embedding[token, :]
        output = self.gru(token_emb.transpose(1, 0), self.initHidden())
        return output

    def initHidden(self):
        return torch.zeros(
            self.state_num, self.batch_size, self.hidden_size, device=self.device
        )


class LocPredGruModel(Module):
    def __init__(
        self, hparams, lane_feature, type_feature, length_feature, node_feature
    ):
        super(LocPredGruModel, self).__init__()
        self.hparams = hparams

        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(
            self.hparams.device
        )
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(
            self.hparams.device
        )
        self.length_emb_layer = nn.Embedding(
            hparams.length_num, hparams.length_dims
        ).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(
            self.hparams.device
        )
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        self.raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)

        self.gru = nn.GRU(hparams.hidden_dims, hparams.hidden_dims)
        self.linear = torch.nn.Linear(hparams.hidden_dims, hparams.node_num)

    def forward(self, input_bat):  # batch_size * length * dims
        init_hidden = torch.zeros(
            1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device
        )
        #    input_emb = self.init_emb[input_bat]
        input_emb = self.raw_feat[input_bat]  # self.gru_embedding(input_bat)

        output_state, _ = self.gru(
            input_emb.view(input_emb.shape[1], input_emb.shape[0], input_emb.shape[2]),
            init_hidden,
        )

        pred_tra = self.linear(output_state)

        return pred_tra


class LocPredGcnModel(Module):
    def __init__(
        self, hparams, lane_feature, type_feature, length_feature, node_feature, adj
    ):
        super(LocPredGcnModel, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()

        self.graph_enc = GcnEncoder(hparams)
        self.lane_feature = lane_feature
        self.type_feature = type_feature
        self.length_feature = length_feature
        self.node_feature = node_feature
        self.adj = adj
        self.gru = nn.GRU(hparams.hidden_dims * 1, hparams.hidden_dims)
        self.linear = torch.nn.Linear(hparams.hidden_dims * 2, hparams.node_num)
        self.linear_red_dim = torch.nn.Linear(hparams.hidden_dims, 50)

    def forward(self, input_bat):  # batch_size * length * dims
        init_hidden = torch.zeros(
            1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device
        )
        self.node_emb = self.graph_enc(
            self.node_feature,
            self.type_feature,
            self.length_feature,
            self.lane_feature,
            self.adj,
        )
        input_emb = self.graph_enc.init_feat[input_bat]
        gcn_emb = self.node_emb[input_bat]  # self.gru_embedding(input_bat)
        #    input_emb = input_emb + gcn_emb
        #    self.init_emb = self.graph_enc.init_feat
        #    self.out_node_emb = self.linear_red_dim(self.node_emb)
        #    input_emb = self.init_emb[input_bat]
        #    input_emb = torch.cat((self.out_node_emb[input_bat], self.init_emb[input_bat]), 2)  # self.gru_embedding(input_bat)

        output_state, _ = self.gru(
            input_emb.view(input_emb.shape[1], input_emb.shape[0], input_emb.shape[2]),
            init_hidden,
        )

        output_state = torch.cat(
            (
                output_state,
                gcn_emb.view(gcn_emb.shape[1], gcn_emb.shape[0], gcn_emb.shape[2]),
            ),
            2,
        )

        pred_tra = self.linear(output_state)

        return pred_tra


class LocPredGatModel(Module):
    def __init__(
        self, hparams, lane_feature, type_feature, length_feature, node_feature, adj
    ):
        super(LocPredGatModel, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()

        self.graph_enc = GatEncoder(hparams)

        self.node_emb = self.graph_enc(
            node_feature, type_feature, length_feature, lane_feature, adj
        )
        self.gru = nn.GRU(hparams.hidden_dims * 1, hparams.hidden_dims)
        self.linear = torch.nn.Linear(hparams.hidden_dims * 2, hparams.node_num)
        self.linear_red_dim = torch.nn.Linear(hparams.hidden_dims, 100)

    def forward(self, input_bat):  # batch_size * length * dims
        init_hidden = torch.zeros(
            1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device
        )
        input_emb = self.graph_enc.init_feat[input_bat]
        gat_emb = self.node_emb[input_bat]  # self.gru_embedding(input_bat)

        #    self.init_emb = self.graph_enc.init_feat
        #    self.out_node_emb = self.linear_red_dim(self.node_emb)
        #    input_emb = self.init_emb[input_bat]
        #    input_emb = torch.cat((self.out_node_emb[input_bat], self.init_emb[input_bat]), 2)  # self.gru_embedding(input_bat)

        output_state, _ = self.gru(
            input_emb.view(input_emb.shape[1], input_emb.shape[0], input_emb.shape[2]),
            init_hidden,
        )

        output_state = torch.cat(
            (
                output_state,
                gat_emb.view(gat_emb.shape[1], gat_emb.shape[0], gat_emb.shape[2]),
            ),
            2,
        )

        pred_tra = self.linear(output_state)

        return pred_tra


class GcnEncoder(Module):
    def __init__(self, hparams):
        super(GcnEncoder, self).__init__()
        self.hparams = hparams

        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(
            self.hparams.device
        )
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(
            self.hparams.device
        )
        self.length_emb_layer = nn.Embedding(
            hparams.length_num, hparams.length_dims
        ).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(
            self.hparams.device
        )

        self.gcn_layer = SPGCN(
            in_features=self.hparams.hidden_dims,
            out_features=int(self.hparams.hidden_dims),
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.init_feat = None

    def forward(self, node_feature, type_feature, length_feature, lane_feature, adj):
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        self.init_feat = raw_feat

        for i in range(self.hparams.baseline_gat_layer):
            raw_feat = self.gcn_layer(raw_feat, adj)
        return raw_feat


class GatEncoder(Module):
    def __init__(self, hparams):
        super(GatEncoder, self).__init__()
        self.hparams = hparams

        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(
            self.hparams.device
        )
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(
            self.hparams.device
        )
        self.length_emb_layer = nn.Embedding(
            hparams.length_num, hparams.length_dims
        ).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(
            self.hparams.device
        )

        self.gat_layer = SPGAT(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            dropout=self.hparams.dropout,
            alpha=self.hparams.alpha,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.init_feat = None

    def forward(self, node_feature, type_feature, length_feature, lane_feature, adj):
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        self.init_feat = raw_feat

        for i in range(self.hparams.baseline_gat_layer):
            raw_feat = self.gat_layer(raw_feat, adj)

        return raw_feat


class LabelPredGatModel(Module):
    def __init__(
        self,
        hparams,
        lane_feature,
        type_feature,
        length_feature,
        node_feature,
        adj,
        struct_assign,
        fnc_assign,
    ):
        super(LabelPredGatModel, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()

        edge = adj._indices().to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
        struct_inter = self.special_spmm(
            edge, edge_e, torch.Size([adj.shape[0], adj.shape[1]]), struct_assign
        )  # N*N   N*C
        struct_adj = torch.mm(struct_assign.t(), struct_inter)  # get struct_adj

        self.graph_enc = GatEncoder(hparams)

        self.node_emb = self.graph_enc(
            node_feature, type_feature, length_feature, lane_feature, adj
        )

        self.linear = torch.nn.Linear(hparams.hidden_dims, hparams.label_num)

        self.linear_red_dim = torch.nn.Linear(hparams.hidden_dims, 100)

    def forward(self, input_bat):  # batch_size * length * dims
        self.init_emb = self.graph_enc.init_feat
        #    input_emb = self.init_emb[input_bat]

        #    input_emb = torch.cat((self.out_node_emb[input_bat], self.init_emb[input_bat]), 2)  # self.gru_embedding(input_bat)

        output_state = self.node_emb[input_bat]
        pred_tra = self.linear(output_state)

        return pred_tra


class LabelPredGcnModel(Module):
    def __init__(
        self,
        hparams,
        lane_feature,
        type_feature,
        length_feature,
        node_feature,
        adj,
        struct_assign,
        fnc_assign,
    ):
        super(LabelPredGcnModel, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()

        edge = adj._indices().to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
        struct_inter = self.special_spmm(
            edge, edge_e, torch.Size([adj.shape[0], adj.shape[1]]), struct_assign
        )  # N*N   N*C
        struct_adj = torch.mm(struct_assign.t(), struct_inter)  # get struct_adj

        self.graph_enc = GcnEncoder(hparams)

        self.node_emb = self.graph_enc(
            node_feature, type_feature, length_feature, lane_feature, adj
        )

        self.linear = torch.nn.Linear(hparams.hidden_dims * 2, hparams.label_num)

        self.linear_red_dim = torch.nn.Linear(hparams.hidden_dims, 100)

    def forward(self, input_bat):  # batch_size * length * dims
        self.init_emb = self.graph_enc.init_feat
        #    input_emb = self.init_emb[input_bat]

        #    input_emb = torch.cat((self.out_node_emb[input_bat], self.init_emb[input_bat]), 2)  # self.gru_embedding(input_bat)
        output_state = torch.cat(
            (self.node_emb[input_bat], self.init_emb[input_bat]), 1
        )

        #    output_state = self.node_emb[input_bat]

        pred_tra = self.linear(output_state)

        return pred_tra


class LabelPredModel(Module):
    def __init__(
        self,
        hparams,
        lane_feature,
        type_feature,
        length_feature,
        node_feature,
        adj,
        struct_assign,
        fnc_assign,
    ):
        super(LabelPredModel, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()
        self.lane_feature = lane_feature
        self.type_feature = type_feature
        self.length_feature = length_feature
        self.node_feature = node_feature
        self.adj = adj
        edge = adj._indices().to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
        struct_inter = self.special_spmm(
            edge, edge_e, torch.Size([adj.shape[0], adj.shape[1]]), struct_assign
        )  # N*N   N*C
        struct_adj = torch.mm(struct_assign.t(), struct_inter)  # get struct_adj

        self.graph_enc = GraphEncoderTL(hparams, struct_assign, fnc_assign, struct_adj)

        self.node_emb = self.graph_enc(
            node_feature, type_feature, length_feature, lane_feature, adj
        )

        self.linear = torch.nn.Linear(hparams.hidden_dims * 2, hparams.label_num)

        self.linear_red_dim = torch.nn.Linear(hparams.hidden_dims, 100)

    def forward(self, input_bat):  # batch_size * length * dims
        init_hidden = torch.zeros(
            1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device
        )
        self.node_emb = self.graph_enc(
            self.node_feature,
            self.type_feature,
            self.length_feature,
            self.lane_feature,
            self.adj,
        )
        self.init_emb = self.graph_enc.init_feat
        #    input_emb = self.init_emb[input_bat]

        #    input_emb = torch.cat((self.out_node_emb[input_bat], self.init_emb[input_bat]), 2)  # self.gru_embedding(input_bat)

        output_state = torch.cat(
            (self.node_emb[input_bat], self.init_emb[input_bat]), 1
        )
        #    output_state = self.node_emb[input_bat]
        #    output_state = self.node_emb[input_bat]
        #    output_state = self.init_emb[input_bat]
        pred_tra = self.linear(output_state)

        return pred_tra


class RoutePlanModel(Module):
    def __init__(
        self,
        hparams,
        lane_feature,
        type_feature,
        length_feature,
        node_feature,
        adj,
        struct_assign,
        fnc_assign,
    ):
        super(RoutePlanModel, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()

        edge = adj._indices().to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
        struct_inter = self.special_spmm(
            edge, edge_e, torch.Size([adj.shape[0], adj.shape[1]]), struct_assign
        )  # N*N   N*C
        struct_adj = torch.mm(struct_assign.t(), struct_inter)  # get struct_adj

        self.graph_enc = GraphEncoderTL(hparams, struct_assign, fnc_assign, struct_adj)

        self.node_emb = self.graph_enc(
            node_feature, type_feature, length_feature, lane_feature, adj
        )

        self.gru = nn.GRU(hparams.hidden_dims * 1, hparams.hidden_dims)

        self.linear = torch.nn.Linear(hparams.hidden_dims, hparams.node_num)

        self.linear_red_dim = torch.nn.Linear(hparams.hidden_dims, 100)

    def forward(self, input_bat, des):  # batch_size * length * dims
        init_hidden = torch.zeros(
            1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device
        )
        self.init_emb = self.graph_enc.init_feat
        #    input_emb = self.init_emb[input_bat]
        input_emb = self.node_emb[input_bat]
        des_emb = self.node_emb[des]
        output_state, _ = self.gru(
            input_emb.view(input_emb.shape[1], input_emb.shape[0], input_emb.shape[2]),
            init_hidden,
        )
        output_state = output_state + des_emb
        pred_tra = self.linear(output_state)

        return pred_tra


class LocPredModel(Module):
    def __init__(
        self,
        hparams,
        lane_feature,
        type_feature,
        length_feature,
        node_feature,
        adj,
        struct_assign,
        fnc_assign,
    ):
        super(LocPredModel, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()
        self.node_feature = node_feature
        self.type_feature = type_feature
        self.length_feature = length_feature
        self.lane_feature = lane_feature
        self.adj = adj
        edge = adj._indices().to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
        struct_inter = self.special_spmm(
            edge, edge_e, torch.Size([adj.shape[0], adj.shape[1]]), struct_assign
        )  # N*N   N*C
        struct_adj = torch.mm(struct_assign.t(), struct_inter)  # get struct_adj

        self.graph_enc = GraphEncoderTL(hparams, struct_assign, fnc_assign, struct_adj)

        self.node_emb = self.graph_enc(
            node_feature, type_feature, length_feature, lane_feature, adj
        )

        self.gru = nn.GRU(hparams.hidden_dims * 1, hparams.hidden_dims)

        self.linear = torch.nn.Linear(hparams.hidden_dims * 2, hparams.node_num)

        self.linear_red_dim = torch.nn.Linear(hparams.hidden_dims, 100)

    def forward(self, input_bat):  # batch_size * length * dims
        init_hidden = torch.zeros(
            1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device
        )
        self.node_emb = self.graph_enc(
            self.node_feature,
            self.type_feature,
            self.length_feature,
            self.lane_feature,
            self.adj,
        )
        self.init_emb = self.graph_enc.init_feat
        #    self.out_node_emb = self.linear_red_dim(self.node_emb)
        input_emb = self.init_emb[input_bat]
        #    input_emb = torch.cat((self.node_emb[input_bat], self.init_emb[input_bat]), 2)  # self.gru_embedding(input_bat)

        output_state, _ = self.gru(
            input_emb.view(input_emb.shape[1], input_emb.shape[0], input_emb.shape[2]),
            init_hidden,
        )
        pred_tra = self.linear(
            torch.cat(
                (
                    output_state,
                    self.node_emb[input_bat].view(
                        self.node_emb[input_bat].shape[1],
                        self.node_emb[input_bat].shape[0],
                        self.node_emb[input_bat].shape[2],
                    ),
                ),
                2,
            )
        )
        #    pred_tra = self.linear(output_state)

        return pred_tra


class GraphEncoderTL(Module):
    def __init__(self, hparams, struct_assign, fnc_assign, struct_adj):
        super(GraphEncoderTL, self).__init__()
        self.hparams = hparams
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign
        self.struct_adj = struct_adj

        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(
            self.hparams.device
        )
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(
            self.hparams.device
        )
        self.length_emb_layer = nn.Embedding(
            hparams.length_num, hparams.length_dims
        ).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(
            self.hparams.device
        )

        self.tl_layer = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign)

        self.init_feat = None

    def forward(self, node_feature, type_feature, length_feature, lane_feature, adj):
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        self.init_feat = raw_feat

        for i in range(self.hparams.loc_pred_gnn_layer):
            raw_feat = self.tl_layer(self.struct_adj, raw_feat, adj)

        return raw_feat


class GraphEncoderGcnCore(Module):
    def __init__(self, hparams, struct_assign, fnc_assign):
        super(GraphEncoderGcnCore, self).__init__()
        self.hparams = hparams
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign

        self.fnc_gcn = GraphConvolution(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.struct_gcn = GraphConvolution(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.node_gat = SPGCN(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            #        dropout = self.hparams.dropout,
            #        alpha = self.hparams.alpha,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.n2s_gcn = SPGCN(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.struct_cmt_num,
            #        dropout = self.hparams.dropout,
            #        alpha = self.hparams.alpha,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.l_c = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.hparams.device)

        self.l_s = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.hparams.device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, struct_adj, raw_feat, raw_adj):

        self.struct_assign = self.n2s_gcn(raw_feat, raw_adj)
        self.struct_assign = F.softmax(self.struct_assign, 0)

        # forward
        self.raw_struct_assign = self.struct_assign
        self.raw_fnc_assign = self.fnc_assign

        self.fnc_assign = self.fnc_assign / (
            F.relu(torch.sum(self.fnc_assign, 0) - 1.0) + 1.0
        )

        self.struct_emb = torch.mm(self.struct_assign.t(), raw_feat)
        self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)
        # backward
        ## F2F
        self.fnc_adj = F.sigmoid(torch.mm(self.fnc_emb, self.fnc_emb.t()))  # n_f * n_f
        self.fnc_adj = (
            self.fnc_adj
            + torch.eye(self.fnc_adj.shape[0]).to(self.hparams.device) * 1.0
        )
        self.fnc_emb = self.fnc_gcn(
            self.fnc_emb.unsqueeze(0), self.fnc_adj.unsqueeze(0)
        ).squeeze()

        ## F2C
        fnc_message = torch.div(
            torch.mm(self.raw_fnc_assign, self.fnc_emb),
            (F.relu(torch.sum(self.fnc_assign, 1) - 1.0) + 1.0).unsqueeze(1),
        )

        self.r_f = self.sigmoid(self.l_c(torch.cat((self.struct_emb, fnc_message), 1)))
        self.struct_emb = self.struct_emb + 0.0 * fnc_message  # magic number: 0.15

        ## C2C
        struct_adj = (
            F.relu(
                struct_adj
                - torch.eye(struct_adj.shape[1]).to(self.hparams.device) * 10000.0
            )
            + torch.eye(struct_adj.shape[1]).to(self.hparams.device) * 1.0
        )
        self.struct_emb = self.struct_gcn(
            self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)
        ).squeeze()

        ## C2N
        struct_message = torch.mm(self.raw_struct_assign, self.struct_emb)
        self.r_s = self.sigmoid(self.l_s(torch.cat((raw_feat, struct_message), 1)))
        raw_feat = raw_feat + struct_message

        #    struct_message = torch.mm(self.raw_struct_assign, self.struct_emb)
        #    raw_feat = raw_feat + 0.0 * struct_message
        ## N2N
        raw_feat = self.node_gat(raw_feat, raw_adj)

        return raw_feat


class GraphEncoderTLCore(Module):
    def __init__(self, hparams, struct_assign, fnc_assign):
        super(GraphEncoderTLCore, self).__init__()
        self.hparams = hparams
        self.raw_struct_assign = struct_assign
        self.raw_fnc_assign = fnc_assign

        self.fnc_gcn = GraphConvolution(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.struct_gcn = GraphConvolution(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.node_gat = SPGCN(
            in_features=self.hparams.hidden_dims,
            out_features=self.hparams.hidden_dims,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.l_c = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.hparams.device)

        self.l_s = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.hparams.device)

        self.l_i = torch.nn.Linear(hparams.hidden_dims, hparams.hidden_dims).to(
            self.hparams.device
        )

        self.sigmoid = nn.Sigmoid()

        self.batch_norm_1 = nn.BatchNorm1d(hparams.hidden_dims, eps=1e-12).cuda()
        self.batch_norm_2 = nn.BatchNorm1d(hparams.hidden_dims, eps=1e-12).cuda()

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
                struct_adj
                - torch.eye(struct_adj.shape[1]).to(self.hparams.device) * 10000.0
            )
            + torch.eye(struct_adj.shape[1]).to(self.hparams.device) * 1.0
        )
        self.struct_emb = self.struct_gcn(
            self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)
        ).squeeze()

        struct_message = torch.mm(self.raw_struct_assign, self.struct_emb)
        self.r_s = self.sigmoid(self.l_s(torch.cat((raw_feat, struct_message), 1)))
        raw_feat = self.node_gat(raw_feat, raw_adj)
        #    raw_feat = raw_feat + self.r_s * 0.7 * struct_message
        return raw_feat


# class GraphEncoderTLCore(Module):
#  def __init__(self, hparams, struct_assign, fnc_assign):
#    super(GraphEncoderTLCore, self).__init__()
#    self.hparams = hparams
#    self.struct_assign = struct_assign
#    self.fnc_assign = fnc_assign

#    self.fnc_gcn = GraphConvolution(
#        in_features = self.hparams.hidden_dims,
#        out_features = self.hparams.hidden_dims,
#        device = self.hparams.device).to(self.hparams.device)

#    self.struct_gcn = GraphConvolution(
#        in_features = self.hparams.hidden_dims,
#        out_features = self.hparams.hidden_dims,
#        device = self.hparams.device).to(self.hparams.device)

#    self.node_gat = SPGCN(
#        in_features = self.hparams.hidden_dims,
#        out_features = self.hparams.hidden_dims,
#        device = self.hparams.device).to(self.hparams.device)


#    self.l_c = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.hparams.device)

#    self.l_s = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.hparams.device)

#    self.sigmoid = nn.Sigmoid()

#  def forward(self, struct_adj, raw_feat, raw_adj):

# forward
#    self.raw_struct_assign = self.struct_assign
#    self.raw_fnc_assign = self.fnc_assign

#    self.struct_assign = self.struct_assign / (F.relu(torch.sum(self.struct_assign, 0) - 1.0) + 1.0)
#    self.fnc_assign = self.fnc_assign / (F.relu(torch.sum(self.fnc_assign, 0) - 1.0) + 1.0)

#    self.struct_emb = torch.mm(self.struct_assign.t(), raw_feat)
#    self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)
# backward
## F2F
#    self.fnc_adj = F.sigmoid(torch.mm(self.fnc_emb, self.fnc_emb.t()))        #n_f * n_f
#    self.fnc_adj = self.fnc_adj + torch.eye(self.fnc_adj.shape[0]).to(self.hparams.device) * 1.0
#    self.fnc_emb = self.fnc_gcn(self.fnc_emb.unsqueeze(0), self.fnc_adj.unsqueeze(0)).squeeze()

## F2C
#    fnc_message = torch.div(torch.mm(self.raw_fnc_assign, self.fnc_emb), (F.relu(torch.sum(self.fnc_assign, 1) - 1.0) + 1.0).unsqueeze(1))

#    self.r_f = self.sigmoid(self.l_c(torch.cat((self.struct_emb, fnc_message), 1)))
#    self.struct_emb = self.struct_emb + 0.15 * fnc_message  #magic number: 0.15

## C2C
#    struct_adj = F.relu(struct_adj - torch.eye(struct_adj.shape[1]).to(self.hparams.device) * 10000.0) + torch.eye(struct_adj.shape[1]).to(self.hparams.device) * 1.0
#    self.struct_emb = self.struct_gcn(self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)).squeeze()

## C2N
#    struct_message = torch.mm(self.raw_struct_assign, self.struct_emb)
#    self.r_s= self.sigmoid(self.l_s(torch.cat((raw_feat, struct_message), 1)))
#    raw_feat = raw_feat + 0.3 * struct_message

## N2N
#    raw_feat = self.node_gat(raw_feat, raw_adj)

#    return raw_feat


class GraphEncoder(Module):
    def __init__(self, hparams):
        super(GraphEncoder, self).__init__()
        self.hparams = hparams
        self.node_num = self.hparams.node_num
        self.cmt_num = self.hparams.cmt_num
        self.cmt_weight = Parameter(torch.FloatTensor(self.node_num, self.cmt_num))
        torch.nn.init.xavier_uniform(self.cmt_weight)

        self.gnn_layers = hparams.gnn_layers
        self.graph_encoders = []
        for i in range(self.gnn_layers):
            if hparams.use_cn_gnn:
                graph_encoder = CommunityNodeGNN(hparams).to(self.hparams.device)
            else:
                graph_encoder = PlainGCN(hparams).to(self.hparams.device)
            self.graph_encoders.append(graph_encoder)

    def forward(self, inputs, adj):  # one layer CN-GNN
        for graph_encoder in self.graph_encoders:
            inputs = graph_encoder(inputs, adj, self.cmt_weight)
        return inputs


class PlainGCN(Module):
    def __init__(self, hparams):
        super(PlainGCN, self).__init__()
        self.hparams = hparams

        #    self.plain_conv = GraphConvolution(
        #        in_features=self.hparams.node_dims,
        #        out_features=self.hparams.node_dims,
        #        device=self.hparams.device).to(self.hparams.device)

        self.plain_conv = SPGAT(
            in_features=self.hparams.node_dims,
            out_features=self.hparams.node_dims,
            dropout=self.hparams.dropout,
            alpha=self.hparams.alpha,
            device=self.hparams.device,
        ).to(self.hparams.device)

    def forward(self, inputs, adj, cmt_weight):  # one layer CN-GNN
        node_out = F.relu(
            self.plain_conv(inputs.unsqueeze(0), adj.unsqueeze(0).float())
        ).squeeze()
        return node_out


class GraphAutoencoderTra(Module):
    def __init__(self, hparams):
        super(GraphAutoencoderTra, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()
        self.fnc_cmt_gat = GraphConvolution(
            in_features=self.hparams.struct_cmt_dims,
            out_features=self.hparams.fnc_cmt_num,
            device=self.hparams.device,
        ).to(self.hparams.device)
        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(
            self.hparams.device
        )
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(
            self.hparams.device
        )
        self.length_emb_layer = nn.Embedding(
            hparams.length_num, hparams.length_dims
        ).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(
            self.hparams.device
        )
        self.linear = torch.nn.Linear(hparams.hidden_dims * 3, 100)

    def forward(
        self,
        lane_feature,
        type_feature,
        length_feature,
        node_feature,
        raw_adj,
        t_adj,
        struct_assign,
        s_edge,
    ):
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        self.raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        self.struct_assign = struct_assign / (
            F.relu(torch.sum(struct_assign, 0) - 1.0) + 1.0
        )
        #    self.struct_assign = F.softmax(struct_assign, 0)

        self.struct_emb = torch.mm(self.struct_assign.t(), self.raw_feat)
        edge = raw_adj._indices()  # .to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
        struct_inter = self.special_spmm(
            edge,
            edge_e,
            torch.Size([raw_adj.shape[0], raw_adj.shape[1]]),
            self.struct_assign,
        )  # N*N   N*C

        #    print("struct_inter:", struct_inter)
        temp = torch.sparse_coo_tensor(
            edge, edge_e, torch.Size([raw_adj.shape[0], raw_adj.shape[1]])
        )
        #    print("temp:", temp)
        #    print("struct_assign", stuct_assign)
        #    print("struct_inter dense:", torch.mm(torch.sparse_coo_tensor(edge, edge_e, torch.Size([raw_adj.shape[0], raw_adj.shape[1]])).to_dense(), self.struct_assign))
        #    print(torch.sparse_coo_tensor(edge, edge_e, torch.Size([raw_adj.shape[0], raw_adj.shape[1]])).to_dense()[0, :], self.struct_assign[:, 0])
        struct_adj = torch.mm(self.struct_assign.t(), struct_inter)
        # print("struct_assign:", struct_assign[0, :], struct_assign[10, :], struct_assign[100, :])
        self.fnc_assign = self.fnc_cmt_gat(
            self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)
        ).squeeze()
        #    print(self.fnc_assign[:20, :])
        self.fnc_assign = F.softmax(self.fnc_assign, 0)
        self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)

        N_C = torch.mm(self.struct_assign, self.struct_emb)
        N_F = torch.mm(torch.mm(self.struct_assign, self.fnc_assign), self.fnc_emb)
        X = torch.cat([self.raw_feat, N_C, N_F], 1)
        X = self.linear(X)

        #    pred_edge = self.linear(torch.cat([X[s_edge[0, :], :], X[s_edge[1, :]], 1))
        pred_edge = torch.einsum("ij,ij->i", X[s_edge[0, :], :], X[s_edge[1, :], :])

        return pred_edge


class GraphAutoencoder(Module):
    def __init__(self, hparams):
        super(GraphAutoencoder, self).__init__()
        self.hparams = hparams
        self.enc_in_dims = (
            self.hparams.node_dims
            + self.hparams.lane_dims
            + self.hparams.type_dims
            + self.hparams.length_dims
        )
        self.enc_out_dims = self.hparams.struct_cmt_num
        self.enc_gnn = StructuralGNN(hparams, self.enc_in_dims, self.enc_out_dims).to(
            self.hparams.device
        )
        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(
            self.hparams.device
        )
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(
            self.hparams.device
        )
        self.length_emb_layer = nn.Embedding(
            hparams.length_num, hparams.length_dims
        ).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(
            self.hparams.device
        )
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * self.enc_in_dims)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(hparams.alpha)
        self.sigmoid = nn.Sigmoid()
        self.dec_gnn = StructuralDecoder(
            hparams, self.enc_in_dims, hparams.node_num
        ).to(self.hparams.device)

    def forward(
        self, main_adj, lane_feature, type_feature, length_feature, node_feature, f_edge
    ):
        # print(
        #     main_adj.shape,
        #     lane_feature.shape,
        #     type_feature.shape,
        #     length_feature.shape,
        #     node_feature.shape,
        # )
        # print(length_feature, self.length_emb_layer.shape)
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        main_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        struct_emb, struct_adj, main_assign = self.enc_gnn(main_feat, main_adj)
        #    print("struct_emb:", struct_emb.shape, struct_emb)
        res_emb = self.dec_gnn(struct_emb, struct_adj)
        #    print("res_emb:", res_emb.shape, res_emb, torch.sum(res_emb, 1))
        t_edge = main_adj._indices().to(self.hparams.device)
        edge = torch.cat([t_edge, f_edge], 1)
        edge_h = torch.cat((res_emb[edge[0, :], :], res_emb[edge[1, :], :]), dim=1).t()
        #    print("eeedge h:", edge_h.shape, edge_h, self.a.mm(edge_h).shape, self.a.mm(edge_h))
        #    edge_e = self.sigmoid(self.leakyrelu(self.a.mm(edge_h).squeeze()))
        edge_e = self.sigmoid(
            torch.einsum("ij,ij->i", res_emb[edge[0, :], :], res_emb[edge[1, :], :])
        )
        edge_label = torch.cat(
            [
                torch.ones(t_edge.shape[1], dtype=torch.float),
                torch.zeros(t_edge.shape[1], dtype=torch.float),
            ]
        ).to(self.hparams.device)

        pred_cmt_adj = torch.mm(struct_emb, struct_emb.t())
        return edge_h, struct_adj, pred_cmt_adj, main_assign, edge_e, edge_label


class Graph2SeqLoc(Module):
    def __init__(
        self,
        hparams,
        lane_feature,
        type_feature,
        length_feature,
        node_feature,
        raw_adj,
        struct_assign,
    ):
        super(Graph2SeqLoc, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()

        self.fnc_cmt_gat = GraphConvolution(
            in_features=self.hparams.struct_cmt_dims,
            out_features=self.hparams.fnc_cmt_num,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(
            self.hparams.device
        )
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(
            self.hparams.device
        )
        self.length_emb_layer = nn.Embedding(
            hparams.length_num, hparams.length_dims
        ).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(
            self.hparams.device
        )

        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        self.raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)

        self.struct_assign = F.softmax(struct_assign, 0)
        self.struct_emb = torch.mm(self.struct_assign.t(), self.raw_feat)

        edge = raw_adj._indices()  # .to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
        struct_inter = self.special_spmm(
            edge,
            edge_e,
            torch.Size([raw_adj.shape[0], raw_adj.shape[1]]),
            self.struct_assign,
        )  # N*N   N*C
        struct_adj = torch.mm(self.struct_assign.t(), struct_inter)

        self.fnc_assign = self.fnc_cmt_gat(
            self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)
        ).squeeze()

        self.fnc_assign = F.softmax(self.fnc_assign, 0)

        self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)

        self.gru = nn.GRU(hparams.hidden_dims, hparams.hidden_dims)

        self.linear = torch.nn.Linear(hparams.hidden_dims, hparams.node_num)

        self.linears = torch.nn.Linear(hparams.hidden_dims * 1, 50)

        self.linearf = torch.nn.Linear(hparams.hidden_dims * 1, 50)

    def forward(self, input_bat):  # batch_size * length * dims
        init_hidden = torch.zeros(
            1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device
        )
        input_emb = self.raw_feat[input_bat]  # self.gru_embedding(input_bat)

        struct_emb = torch.einsum(
            "ijk,kl->ijl", self.struct_assign[input_bat], self.struct_emb
        )
        fnc_emb = torch.einsum(
            "ijk,kl->ijl",
            torch.einsum("ijk,kl->ijl", self.struct_assign[input_bat], self.fnc_assign),
            self.fnc_emb,
        )

        #    input_emb = torch.cat((input_emb, self.linears(struct_emb), self.linearf(fnc_emb)), 2)
        output_state, _ = self.gru(
            input_emb.view(input_emb.shape[1], input_emb.shape[0], input_emb.shape[2]),
            init_hidden,
        )

        #    output_state = torch.cat((output_state.permute(1, 0, 2), self.linears(struct_emb), self.linearf(fnc_emb)), 2)

        pred_tra = self.linear(output_state)

        return pred_tra


class Graph2SeqCmt(Module):
    def __init__(
        self,
        hparams,
        lane_feature,
        type_feature,
        length_feature,
        node_feature,
        main_adj,
        main_assign,
    ):
        super(Graph2SeqCmt, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()

        self.fnc_cmt_gat = GraphConvolution(
            in_features=self.hparams.struct_cmt_dims,
            out_features=self.hparams.fnc_cmt_num,
            #        dropout = self.hparams.dropout,
            #        alpha = self.hparams.alpha,
            device=self.hparams.device,
        ).to(self.hparams.device)
        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(
            self.hparams.device
        )
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(
            self.hparams.device
        )
        self.length_emb_layer = nn.Embedding(
            hparams.length_num, hparams.length_dims
        ).to(self.hparams.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(
            self.hparams.device
        )
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        main_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)

        main_assign = F.softmax(main_assign, 0)
        self.struct_emb = torch.mm(main_assign.t(), main_feat)

        edge = main_adj._indices()  # .to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)

        struct_inter = self.special_spmm(
            edge,
            edge_e,
            torch.Size([main_adj.shape[0], main_adj.shape[1]]),
            main_assign,
        )  # N*N   N*C
        struct_adj = torch.mm(main_assign.t(), struct_inter)

        self.fnc_assign = self.fnc_cmt_gat(
            self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)
        ).squeeze()

        self.fnc_assign = F.softmax(self.fnc_assign, 0)

        self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)
        self.fnc_adj = torch.mm(
            torch.mm(self.fnc_assign.t(), struct_adj), self.fnc_assign
        )

        self.gru = nn.GRU(hparams.struct_cmt_dims, hparams.struct_cmt_dims)

        self.gru_embedding = nn.Embedding(
            hparams.struct_cmt_num, hparams.hidden_dims
        ).to(self.hparams.device)
        self.linear = torch.nn.Linear(hparams.hidden_dims * 1, hparams.struct_cmt_num)

    def forward(self, input_bat):  # batch_size * length * dims
        init_hidden = torch.zeros(
            1, input_bat.shape[0], self.hparams.hidden_dims, device=self.hparams.device
        )
        input_emb = self.gru_embedding(input_bat)
        input_state, _ = self.gru(
            input_emb.view(input_emb.shape[1], input_emb.shape[0], input_emb.shape[2]),
            init_hidden,
        )
        attn_weight = torch.einsum(
            "ijk,lk->ijl", input_state, self.fnc_emb
        )  # cmt_num * cmt_dims  len * batch_size * dims    ->  len * batch_size * cmt_num

        #    out_state = torch.cat((input_state, torch.einsum("ijk,kl->ijl", attn_weight, self.fnc_emb)), 2)
        #    pred_tra = self.linear(out_state)
        pred_tra = self.linear(input_state)

        #    pred_tra = F.softmax(pred_tra, 2)

        return pred_tra


class StructuralDecoder(Module):
    def __init__(self, hparams, in_dims, out_dims):
        super(StructuralDecoder, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()
        self.cmt_gat_0 = GraphConvolution(
            in_features=in_dims, out_features=out_dims, device=self.hparams.device
        ).to(self.hparams.device)
        self.softmax = torch.nn.Softmax(dim=-1)

    #    self.cmt_gat_1 = GraphConvolution(
    #        in_features = self.hparams.node_dims,
    #        out_features = out_dims,
    #        device = self.hparams.device).to(self.hparams.device)

    def forward(self, main_feat, main_adj):
        main_assign = self.cmt_gat_0(main_feat.unsqueeze(0), main_adj.unsqueeze(0))
        #    print("line 199:", main_inter.shape, main_adj.shape)
        #    main_assign = self.cmt_gat_1(main_inter, main_adj.unsqueeze(0))
        #    zero_pad_1 = torch.zeros(main_assign.shape[0], brh_assign.shape[1])
        #    zero_pad_2 = torch.zeros(brh_assign.shape[0], main_assign.shape[1])
        #    top_assign = torch.cat([main_assign, zero_pad_1], 1)
        #    buttom_assign = torch.cat([zero_pad_2, brh_assign], 1)

        #    struct_assign = torch.cat([top_assign, buttom_assign], 0)

        #    all_feat = torch.cat([main_feat, brh_feat], 0)

        main_assign = main_assign.to(self.hparams.device)
        main_assign = main_assign.squeeze()
        #    print("main assign:", main_assign.shape, main_assign)
        main_assign = F.softmax(main_assign, 0)
        #    pickle.dump(main_assign.tolist(), open("struct_assign", "wb"))
        raw_emb = torch.mm(main_assign.t(), main_feat)
        return raw_emb


class StructuralGNN(Module):
    def __init__(self, hparams, in_dims, out_dims):
        super(StructuralGNN, self).__init__()
        self.hparams = hparams
        self.special_spmm = SpecialSpmm()
        self.cmt_gat_0 = SPGAT(
            in_features=in_dims,
            out_features=out_dims,
            dropout=self.hparams.dropout,
            alpha=self.hparams.alpha,
            device=self.hparams.device,
        ).to(self.hparams.device)

    #    self.cmt_gat_1 = SPGAT(
    #        in_features = self.hparams.node_dims,
    #        out_features = out_dims,
    #        dropout = self.hparams.dropout,
    #        alpha = self.hparams.alpha,
    #        device = self.hparams.device).to(self.hparams.device)

    def forward(self, main_feat, main_adj):
        main_assign = self.cmt_gat_0(main_feat, main_adj)
        #    main_assign = self.cmt_gat_1(main_inter, main_adj)
        #    zero_pad_1 = torch.zeros(main_assign.shape[0], brh_assign.shape[1])
        #    zero_pad_2 = torch.zeros(brh_assign.shape[0], main_assign.shape[1])
        #    top_assign = torch.cat([main_assign, zero_pad_1], 1)
        #    buttom_assign = torch.cat([zero_pad_2, brh_assign], 1)

        #    struct_assign = torch.cat([top_assign, buttom_assign], 0)

        #    all_feat = torch.cat([main_feat, brh_feat], 0)

        main_assign = main_assign.to(self.hparams.device)
        main_assign = F.softmax(main_assign, 0)
        #    pickle.dump(main_assign.tolist(), open("struct_assign", "wb"))
        struct_emb = torch.mm(main_assign.t(), main_feat)
        edge = main_adj._indices().to(self.hparams.device)
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.hparams.device)
        struct_inter = self.special_spmm(
            edge,
            edge_e,
            torch.Size([main_adj.shape[0], main_adj.shape[1]]),
            main_assign,
        )  # N*N   N*C
        struct_adj = torch.mm(main_assign.t(), struct_inter)
        struct_adj = F.relu(struct_adj - 0.0001)
        return struct_emb, struct_adj, main_assign


class CommunityNodeGNN(Module):
    def __init__(self, hparams):
        super(CommunityNodeGNN, self).__init__()
        self.hparams = hparams

        self.plain_node_conv = GraphConvolution(
            in_features=self.hparams.node_dims + self.hparams.cmt_dims,
            out_features=self.hparams.node_dims,
            device=self.hparams.device,
        ).to(self.hparams.device)

        self.plain_cmt_conv = GraphConvolution(
            in_features=self.hparams.cmt_dims,
            out_features=self.hparams.cmt_dims,
            device=self.hparams.device,
        ).to(self.hparams.device)

    def forward(self, inputs, adj, cmt_weight):  # one layer CN-GNN
        cmt_weight_softmax = F.softmax(cmt_weight, dim=-1)
        cmt_emb, cmt_adj, _, _ = get_cmt(inputs, adj.float(), cmt_weight_softmax)
        cmt_emb = self.cmt_forward(cmt_emb, cmt_adj)
        inputs = self.node_forward(inputs, adj.float(), cmt_emb, cmt_weight_softmax)
        return inputs

    def node_forward(
        self, inputs, adj, cmt_emb, cmt_weight_softmax, embedding_mask=None
    ):
        #    node_cmt_emb =
        node2cmt = torch.argmax(cmt_weight_softmax, -1)  # node_num * 1
        cmt_emb = cmt_emb.squeeze()
        inputs_cmt_emb = cmt_emb[node2cmt, :]
        inputs_cat = torch.cat([inputs, inputs_cmt_emb], -1)
        node_out = F.relu(
            self.plain_node_conv(inputs_cat.unsqueeze(0), adj.unsqueeze(0))
        ).squeeze()
        return node_out

    def cmt_forward(
        self, cmt_emb, cmt_adj, embedding_mask=None
    ):  # apply plain gcn on cmt level
        cmt_out = F.relu(self.plain_cmt_conv(cmt_emb, cmt_adj)).squeeze()
        if embedding_mask is not None:
            cmt_out = cmt_out * embedding_mask
        return cmt_out
