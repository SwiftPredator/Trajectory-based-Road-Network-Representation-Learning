import json
import os
from itertools import chain, combinations
from platform import node
from turtle import pos

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from mxnet import autograd, gpu, nd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L2Loss, Loss
from mxnet.gluon.nn import ELU
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from models.rfn_original.factory_functions import (
    FeatureInfo,
    RFNLayerSpecification,
    make_rfn,
)
from models.rfn_original.relational_fusion.normalizers import (
    L2Normalization,
    NoNormalization,
)
from models.rfn_original.utils import generate_required_city_graph

from .model import Model


class RFNModel(Model):
    """
    This class is an adapter over the original implementation in mxnet.
    It uses mxnet to build the model.
    """

    def __init__(
        self, data, device, network, emb_dim: int = 128, remove_highway_label=False
    ):
        self.loss_func = ReconLoss()
        self.optimizer = "adam"
        self.lr = 0.001
        self.emb_dim = emb_dim

        self.city = generate_required_city_graph(
            "Porto", network, remove_highway_label=remove_highway_label
        )
        self._build_model()

    def _build_model(self):
        X_V = self.city.X_V
        X_E = self.city.X_E
        X_B = self.city.X_B
        input_feature_info = FeatureInfo.from_feature_matrices(X_V, X_E, X_B)
        print(input_feature_info)
        no_hidden_layers = 2
        hidden_layer_specs = [
            RFNLayerSpecification(
                units=2 * self.emb_dim,
                fusion="interactional",
                aggregator="attentional",
                normalization=L2Normalization(),
                activation=ELU(),
            )
            for _ in range(no_hidden_layers)
        ]
        output_layer_spec = RFNLayerSpecification(
            units=self.emb_dim,
            fusion="additive",
            aggregator="non-attentional",
            normalization=NoNormalization(),
            activation="relu",
        )
        self.rfn = make_rfn(
            input_feature_info,
            hidden_layer_specs,
            output_layer_spec,
            output_mode="edge",
        )
        self.params = self.initialize()

    def train(self, epochs: int = 30):
        """
        Training method for rfn using an GAE Model with reconstruction loss and rfn as encoder module
        Args:
            epochs (int, optional): _description_. Defaults to 1000.
        """

        trainer = Trainer(self.params, self.optimizer, {"learning_rate": self.lr})
        losses = []
        for epoch in range(epochs):
            with autograd.record():
                z = self.rfn(
                    self.city.X_V,
                    self.city.X_E,
                    self.city.X_B,
                    self.city.N_node_primal,
                    self.city.N_edge_primal,
                    self.city.N_mask_primal,
                    self.city.N_node_dual,
                    self.city.N_edge_dual,
                    self.city.N_common_node,
                    self.city.N_mask_dual,
                )
                loss = self.loss_func(z, self.city.y)
            loss.backward()
            trainer.step(batch_size=len(self.city.y))
            if (epoch + 1) % 100 == 0:
                print(f"Loss at Epoch {epoch+1}: {np.mean(losses)}")
            losses.append(loss.mean().asscalar())
        return losses

    def initialize(self):
        self.rfn.initialize()
        params = self.rfn.collect_params()
        params.reset_ctx(ctx=gpu(1))
        return params

    def save_model(self, path):
        self.rfn.save_parameters(os.path.join(path + "/model.params"))

    def load_model(self, path):
        self.rfn.load_parameters(path)

    def save_emb(self):
        ...

    def load_emb(self):
        z = self.rfn(
            self.city.X_V,
            self.city.X_E,
            self.city.X_B,
            self.city.N_node_primal,
            self.city.N_edge_primal,
            self.city.N_mask_primal,
            self.city.N_node_dual,
            self.city.N_edge_dual,
            self.city.N_common_node,
            self.city.N_mask_dual,
        )

        return z.asnumpy()


class ReconLoss(Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(ReconLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, z, pos_edge_index):
        # calculate inner product (z*z^T)
        val = nd.sum((z[pos_edge_index[0]] * z[pos_edge_index[1]]), axis=1)
        ip = nd.sigmoid(val)
        pos_loss = nd.mean(-nd.log(ip + 1e-15))

        # sample negative edges
        neg_edge_index = self.negative_sample(pos_edge_index)
        neg_val = nd.sum((z[neg_edge_index[0]] * z[neg_edge_index[1]]), axis=1)
        neg_ip = nd.sigmoid(neg_val)
        neg_loss = nd.mean(-nd.log(1 - neg_ip + 1e-15))

        return pos_loss + neg_loss

    def negative_sample(self, pos_edge_idx):
        pos_idx = pos_edge_idx.asnumpy().astype(int)  # (edges, 2)
        neg_idx = negative_sampling(
            torch.tensor(pos_idx, dtype=torch.long), pos_idx.max()
        )
        return nd.array(neg_idx, dtype=int)
