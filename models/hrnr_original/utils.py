import copy
import os
import pickle
import random

import numpy as np
from scipy import sparse

EPS = 1e-30


def gen_false_edge(adj, num):
    #  adj = copy.deepcopy(adj)
    adj = adj.todense()
    edges = []
    while len(edges) < num:
        start = random.randint(0, len(adj) - 1)
        end = random.randint(0, len(adj) - 1)
        if adj[start, end] == 0 and adj[end, start] == 0:
            edges.append([start, end])
    edges = np.array(edges)
    edges = edges.transpose()
    return edges


def generate_edge(t_adj, num):  # generate edge to train g2t model
    false_edges = gen_false_edge(t_adj, num / 2)
    false_edges = false_edges.transpose().tolist()
    true_edges = []
    t_labels = []
    while len(true_edges) < num / 2:
        ind = random.randint(0, len(t_adj.data) - 1)
        start = t_adj.row[ind]
        end = t_adj.col[ind]
        true_edges.append([start, end])
        t_labels.append(t_adj.data[ind])
    edges = false_edges
    edges.extend(true_edges)

    edges = np.array(edges)
    edges = edges.transpose()
    f_labels = [0.0 for i in range(1000)]
    labels = []
    labels.extend(f_labels)
    labels.extend(t_labels)

    return edges, labels


def load_loc_rst_data(hparams):
    adj = pickle.load(open(hparams.adj, "rb"))
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)

    t_adj = pickle.load(open(hparams.t_adj, "rb"))
    t_adj = np.array(t_adj)
    t_adj = t_adj / 1000.0
    t_self_loop = np.eye(len(t_adj))
    t_adj = np.array(t_adj) + t_self_loop
    t_adj = sparse.coo_matrix(t_adj)

    node_features = pickle.load(open(hparams.node_features, "rb"))
    node_features = node_features.tolist()
    while len(node_features) < 16000:
        node_features.append(["0", "0", "0", "0"])
    node_features = np.array(node_features)

    spectral_label = pickle.load(open(hparams.spectral_label, "rb"))

    return adj, node_features, spectral_label, t_adj


def load_g2s_data(hparams):
    adj = pickle.load(open(hparams.adj, "rb"))
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)
    node_features = pickle.load(open(hparams.node_features, "rb"))
    node_features = node_features.tolist()
    while len(node_features) < 16000:
        node_features.append(["0", "0", "0", "0"])
    node_features = np.array(node_features)

    spectral_label = pickle.load(open(hparams.spectral_label, "rb"))
    train_cmt_set = pickle.load(open(hparams.train_cmt_set, "rb"))

    return adj, node_features, spectral_label, train_cmt_set


def get_label_train_data(hparams):
    label_pred_train = pickle.load(open(hparams.label_train_set, "rb"))
    label_train_true = label_pred_train[:-100]
    label_train_false = []
    while len(label_train_false) < len(label_train_true):
        node = random.randint(0, hparams.node_num - 1)
        if (not node in label_train_false) and (not node in label_train_true):
            label_train_false.append(node)

    label_train_set = label_train_false
    label_train_set.extend(label_train_true)
    label_train_real = [0 for i in range(int(len(label_train_set) / 2))]
    label_train_real.extend([1 for i in range(int(len(label_train_set) / 2))])

    label_test_false = pickle.load(open(hparams.label_train_set_false, "rb"))
    label_test_true = label_pred_train[-100:]
    label_test_set = label_test_false
    label_test_set.extend(label_test_true)
    label_test_real = [0 for i in range(100)]
    label_test_real.extend([1 for i in range(100)])

    return label_train_set, label_train_real, label_test_set, label_test_real


def load_label_pred_data(hparams):
    adj = pickle.load(open(hparams.adj, "rb"))
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)
    node_features = pickle.load(open(hparams.node_features, "rb"))
    node_features = node_features.tolist()
    while len(node_features) < 16000:
        node_features.append(["0", "0", "0", "0"])
    node_features = np.array(node_features)

    struct_assign = pickle.load(open(hparams.struct_assign, "rb"))
    fnc_assign = pickle.load(open(hparams.fnc_assign, "rb"))

    return adj, node_features, struct_assign, fnc_assign


def load_des_pred_data(hparams):
    adj = pickle.load(open(hparams.adj, "rb"))
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)
    node_features = pickle.load(open(hparams.node_features, "rb"))
    node_features = node_features.tolist()
    while len(node_features) < 16000:
        node_features.append(["0", "0", "0", "0"])
    node_features = np.array(node_features)

    struct_assign = pickle.load(open(hparams.struct_assign, "rb"))
    fnc_assign = pickle.load(open(hparams.fnc_assign, "rb"))

    train_loc_set = pickle.load(open(hparams.train_loc_set, "rb"))

    return adj, node_features, struct_assign, fnc_assign, train_loc_set


def load_loc_pred_data(hparams):
    adj = pickle.load(open(hparams.adj, "rb"))
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)
    node_features = pickle.load(open(hparams.node_features, "rb"))
    node_features = node_features.tolist()
    while len(node_features) < 16000:
        node_features.append(["0", "0", "0", "0"])
    node_features = np.array(node_features)

    struct_assign = pickle.load(open(hparams.struct_assign, "rb"))
    fnc_assign = pickle.load(open(hparams.fnc_assign, "rb"))

    train_loc_set = pickle.load(open(hparams.train_loc_set, "rb"))

    return adj, node_features, struct_assign, fnc_assign, train_loc_set


def load_g2s_loc_data(hparams):
    adj = pickle.load(open(hparams.adj, "rb"))
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)
    node_features = pickle.load(open(hparams.node_features, "rb"))
    node_features = node_features.tolist()
    while len(node_features) < 16000:
        node_features.append(["0", "0", "0", "0"])
    node_features = np.array(node_features)

    spectral_label = pickle.load(open(hparams.spectral_label, "rb"))
    train_loc_set = pickle.load(open(hparams.train_loc_set, "rb"))

    return adj, node_features, spectral_label, train_loc_set


def load_gae_data(hparams):
    print(os.getcwd())
    adj = pickle.load(open(hparams.adj, "rb"))
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)
    node_features = pickle.load(open(hparams.node_features, "rb"))
    node_features = node_features.tolist()
    while len(node_features) < 16000:
        node_features.append(["0", "0", "0", "0"])
    node_features = np.array(node_features)

    spectral_label = pickle.load(open(hparams.spectral_label, "rb"))

    return adj, node_features, spectral_label


def load_data(hparams):

    train_loc_set = pickle.load(open(hparams.train_loc_set, "rb"))
    train_time_set = pickle.load(open(hparams.train_time_set, "rb"))
    adj = pickle.load(open(hparams.adj, "rb"))
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)
    test_loc_set = train_loc_set[6000:]
    test_time_set = train_time_set[6000:]
    train_loc_set = train_loc_set[:6000]
    train_time_set = train_time_set[:6000]
    return train_loc_set, train_time_set, adj, test_loc_set, test_time_set


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst
