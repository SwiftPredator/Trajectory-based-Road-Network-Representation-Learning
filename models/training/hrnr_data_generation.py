import os
import sys

os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64/"
module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import pandas as pd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import torch
import torch.nn.functional as F
from generator import RoadNetwork, Trajectory
from models.hrnr_original.conf import beijing_hparams

# from models.hrnr_original.train import train_struct_cmt, train_fnc_cmt_rst
from models.hrnr_original.model import *
from models.hrnr_original.utils import *
from networkx import adjacency_matrix, to_numpy_matrix
from scipy import sparse
from sklearn.cluster import SpectralClustering
from torch import optim
from tqdm import tqdm

# Only works for porto atm


def create_adj(network):
    # create needed data to run hrnr encoding model
    adj = to_numpy_matrix(network.line_graph)
    adj = np.hstack([adj, np.zeros(shape=(adj.shape[0], 16000 - adj.shape[1]))])
    adj = np.vstack([adj, np.zeros(shape=(16000 - adj.shape[0], adj.shape[1]))])
    return adj


def create_features(network):
    G = network.line_graph
    idxs = [n for n in G.nodes]
    data = network.gdf_edges.loc[idxs, ["lanes", "highway", "length"]]
    data["id"] = np.arange(data.shape[0])
    data["lanes"] = data["lanes"].str.extract(r"(\w+)")
    data["lanes"].fillna(1, inplace=True)
    data["highway"] = pd.factorize(data["highway"])[0]
    data = data.to_numpy()

    # type_set = pd.Series(data[:, 1])
    # labels, levels = pd.factorize(type_set)
    # data[:, 1] = labels
    # data[:, 2] = (data[:, 2].astype(float) / 0.01).astype(int)
    node_features = data.astype(int)

    return node_features


def create_spectral_cluster(adj):
    sc = SpectralClustering(
        300, affinity="precomputed", n_init=1, assign_labels="discretize"
    )
    sc.fit(adj)
    labels = sc.labels_
    one_hot_label = []
    for item in labels:
        a = [0 for i in range(300)]
        a[item] = 1
        one_hot_label.append(a)

    return np.array(one_hot_label)


def create_tadj(network, trajectory):
    traj = trajectory.df["cpath"]
    skips = 5
    tadj = [[0 for j in range(16000)] for i in range(16000)]
    # create fid to node id mapping
    map = {}
    nodes = list(network.line_graph.nodes)
    for index, id in zip(network.gdf_edges.index, network.gdf_edges.fid):
        map[id] = nodes.index(index)

    for tra in tqdm(traj):
        for i in range(len(tra)):
            for j in range(1, skips + 1):
                if i + j < len(tra):
                    tadj[map[tra[i]]][map[tra[i + j]]] += 1
                    if not tra[i] == tra[i + j]:
                        tadj[map[tra[i + j]]][map[tra[i]]] += 1
    return np.array(tadj)


def get_data(network, trajectory, load_path=None):
    adj = create_adj(network)
    node_features = create_features(network)
    sp_labels = create_spectral_cluster(adj)

    node_features = node_features.tolist()
    while len(node_features) < 16000:
        node_features.append(["0", "0", "0", "0"])
    node_features = np.array(node_features)

    self_loop = np.eye(len(adj))
    adj = np.logical_or(adj, self_loop)
    adj = sparse.coo_matrix(adj, dtype=float)

    t_adj = create_tadj(network, trajectory)
    t_adj = t_adj / 1000.0
    t_self_loop = np.eye(len(t_adj))
    t_adj = np.array(t_adj) + t_self_loop
    t_adj = sparse.coo_matrix(t_adj)

    return adj, node_features, sp_labels, t_adj


def train_struct_cmt_custom(adj, features, sp_label):
    hparams = dict_to_object(beijing_hparams)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # adj, features, sp_label = get_data(network)
    gae_model = GraphAutoencoder(hparams).to(hparams.device)
    mse_criterion = torch.nn.MSELoss()
    ce_criterion = torch.nn.BCELoss()
    model_optimizer = optim.Adam(gae_model.parameters(), lr=hparams.gae_learning_rate)
    adj_indices = torch.tensor(
        np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1),
        dtype=torch.long,
    ).t()
    adj_values = torch.tensor(adj.data, dtype=torch.float)
    adj_shape = adj.shape
    adj_tensor = torch.sparse_coo_tensor(
        adj_indices, adj_values, adj_shape, device=hparams.device
    )
    features = features.astype(np.int)
    lane_feature = torch.tensor(features[:, 0], dtype=torch.long, device=hparams.device)
    type_feature = torch.tensor(features[:, 1], dtype=torch.long, device=hparams.device)
    length_feature = torch.tensor(
        features[:, 2], dtype=torch.long, device=hparams.device
    )
    node_feature = torch.tensor(features[:, 3], dtype=torch.long, device=hparams.device)
    assign_label = torch.tensor(sp_label, dtype=torch.float, device=hparams.device)

    for i in range(hparams.gae_epoch):
        model_optimizer.zero_grad()
        f_edge = gen_false_edge(adj, adj.row.shape[0])
        f_edge = torch.tensor(f_edge, dtype=torch.long, device=hparams.device)
        # print("epoch", i)
        edge_h, struct_adj, pred_cmt_adj, main_assign, edge_e, edge_label = gae_model(
            adj_tensor, lane_feature, type_feature, length_feature, node_feature, f_edge
        )
        #    loss = mse_criterion(struct_adj, pred_cmt_adj)
        # print("edge_e:", torch.mean(edge_e[:10000]), torch.mean(edge_e[-10000:]))
        ce_loss = ce_criterion(edge_e, edge_label)

        loss = ce_criterion(F.softmax(main_assign, 1), assign_label)

        loss.backward(retain_graph=True)
        ce_loss.backward()
        #    print("dec grad:", torch.sum(gae_model.dec_gnn.cmt_gat_0.weight.grad, 1), gae_model.dec_gnn.cmt_gat_0.weight.grad.shape)
        #    print("grad:", gae_model.enc_gnn.cmt_gat_0.a.grad)

        torch.nn.utils.clip_grad_norm_(gae_model.parameters(), hparams.clip)
        model_optimizer.step()
        if i % 50 == 0:
            print(ce_loss.item())
            pickle.dump(main_assign.tolist(), open("struct_assign_porto", "wb"))
            # torch.save(
            #     gae_model.state_dict(),
            #     "/data/wuning/NTLR/beijing/model/gae.model_" + str(i),
            # )


def train_fnc_cmt_rst_custom(
    adj, features, struct_assign, t_adj
):  # train fnc by reconstruction
    hparams = dict_to_object(beijing_hparams)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.device)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # adj, features, struct_assign, t_adj = load_loc_rst_data(hparams)
    ce_criterion = torch.nn.MSELoss()

    adj_indices = torch.tensor(
        np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1),
        dtype=torch.long,
    ).t()
    adj_values = torch.tensor(adj.data, dtype=torch.float)
    adj_shape = adj.shape
    adj_tensor = torch.sparse_coo_tensor(
        adj_indices, adj_values, adj_shape, device=hparams.device
    )

    t_adj_indices = torch.tensor(
        np.concatenate([t_adj.row[:, np.newaxis], t_adj.col[:, np.newaxis]], 1),
        dtype=torch.long,
    ).t()
    t_adj_values = torch.tensor(t_adj.data, dtype=torch.float)
    t_adj_shape = t_adj.shape
    t_adj_tensor = torch.sparse.FloatTensor(t_adj_indices, t_adj_values, t_adj_shape)

    features = features.astype(np.int)

    lane_feature = torch.tensor(features[:, 0], dtype=torch.long, device=hparams.device)
    type_feature = torch.tensor(features[:, 1], dtype=torch.long, device=hparams.device)
    length_feature = torch.tensor(
        features[:, 2], dtype=torch.long, device=hparams.device
    )
    node_feature = torch.tensor(features[:, 3], dtype=torch.long, device=hparams.device)
    struct_assign = torch.tensor(
        struct_assign, dtype=torch.float, device=hparams.device
    )

    g2t_model = GraphAutoencoderTra(hparams).to(hparams.device)

    model_optimizer = optim.Adam(g2t_model.parameters(), lr=hparams.g2t_learning_rate)

    for i in range(hparams.g2t_epoch):
        # print("epoch", i)
        input_edge, label = generate_edge(t_adj, hparams.g2t_sample_num)
        label = torch.tensor(label, dtype=torch.float, device=hparams.device)
        input_edge = torch.tensor(input_edge, dtype=torch.long, device=hparams.device)
        pred = g2t_model(
            lane_feature,
            type_feature,
            length_feature,
            node_feature,
            adj_tensor,
            t_adj_tensor,
            struct_assign,
            input_edge,
        )
        count = 0

        loss = ce_criterion(pred, label)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(g2t_model.parameters(), hparams.g2t_clip)
        model_optimizer.step()
        #      print("grad:", g2s_model.linear.weight.grad)
        if count % 50 == 0:
            print(loss.item())
            # torch.save(
            #     g2t_model.state_dict(),
            #     "/data/wuning/RN-GNN/beijing/model/g2t.model_" + str(i),
            # )
            pickle.dump(
                g2t_model.fnc_assign.tolist(),
                open("fnc_assign_porto", "wb"),
            )
            count += 1
