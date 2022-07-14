import os
from turtle import forward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, GATConv, GCNConv

from .model import Model


class GAEModel(Model):
    def __init__(self, data, device, encoder, emb_dim=128, layers=2):
        self.model = GAE(
            encoder(data.x.shape[1], emb_dim, layers)
        )  # feature dim, emb dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model = self.model.to(device)
        self.device = device
        self.train_data = data
        self.train_data = self.train_data.to(device)

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model.encode(self.train_data.x, self.train_data.edge_index)
            loss = self.model.recon_loss(z, self.train_data.edge_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 500 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

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
            self.model.encode(self.train_data.x, self.train_data.edge_index)
            .detach()
            .cpu()
            .numpy()
        )


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num):
        super().__init__()
        self.layers = nn.Sequential()
        if layer_num == 2:
            self.layers.append(GCNConv(in_channels, 2 * out_channels))
            self.layers.append(GCNConv(2 * out_channels, out_channels))
        else:
            self.layers.append(GCNConv(in_channels, out_channels))

    def forward(self, x, edge_index):
        x = x.float()
        for layer in self.layers[:-1]:
            x = layer(x, edge_index).relu()

        return self.layers[-1](x, edge_index)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num):
        super().__init__()
        self.layers = nn.Sequential()
        if layer_num == 2:
            self.layers.append(GATConv(in_channels, 2 * out_channels))
            self.layers.append(GATConv(2 * out_channels, out_channels))
        else:
            self.layers.append(GATConv(in_channels, out_channels))

    def forward(self, x, edge_index):
        x = x.float()
        for layer in self.layers[:-1]:
            x = layer(x, edge_index).relu()

        return self.layers[-1](x, edge_index)
