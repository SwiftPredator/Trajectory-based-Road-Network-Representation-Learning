import json
import os
from itertools import chain, combinations
from platform import node

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L2Loss, Loss
from mxnet.gluon.nn import ELU
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GAE
from tqdm import tqdm

from rfn_original.factory_functions import FeatureInfo, RFNLayerSpecification, make_rfn
from rfn_original.relational_fusion.normalizers import L2Normalization, NoNormalization

from .model import Model


class RFNModel(Model):
    """
    This class is an adapter over the original implementation in mxnet.
    It uses mxnet to build the model.
    """

    def __init__(self, data, device, network, emb_dim: int = 128):
        self.loss_func = ReconLoss()
        self.optimizer = "adam"
        self.lr = 0.001

    def train(self, epochs: int = 1000):
        """
        Training method for rfn using an GAE Model with reconstruction loss and rfn as encoder module
        Args:
            epochs (int, optional): _description_. Defaults to 1000.
        """

        trainer = Trainer(
            self.params, self.optimizer, {"learning_rate": self.learning_rate}
        )
        losses = []
        for epoch in range(epochs):
            city = self.train_cities[epoch % len(self.train_cities)]
            with autograd.record():
                z = self.rfn(
                    city.X_V,
                    city.X_E,
                    city.X_B,
                    city.N_node_primal,
                    city.N_edge_primal,
                    city.N_mask_primal,
                    city.N_node_dual,
                    city.N_edge_dual,
                    city.N_common_node,
                    city.N_mask_dual,
                )
                loss = self.loss_func(z, pos_index)
            loss.backward()
            trainer.step(batch_size=len(city.y))
            print(f"Loss at Epoch {epoch}: {loss.mean().asscalar()}")
            losses.append(loss.mean().asscalar())
        return losses


class ReconLoss(Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(ReconLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, z, pos_edge_index):
        loss = 0
        return loss
