from turtle import forward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCNConv

from .model import Model


class GAEModel(Model):
    def __init__(self, in_dim, out_dim, device):
        self.model = GAE(GCNEncoder(in_dim, out_dim))  # feature dim, emb dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model = self.model.to(device)
        self.emb = None
        self.device = device

    def train(self, train_data, epochs: int = 10):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model.encode(train_data.x, train_data.edge_index)
            loss = self.model.recon_loss(z, train_data.edge_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 100 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))
        self.emb = (
            self.model.encode(train_data.x, train_data.edge_index)
            .detach()
            .cpu()
            .numpy()
        )

    def save_model(self, save_name, path="save/"):
        torch.save(self.model.state_dict(), path + save_name)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def save_emb(self, path):
        np.savetxt(path + "embedding.out", X=self.emb)

    def load_emb(self, path=None):
        if path:
            self.emb = np.loadtxt(path)
        return self.emb


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index).relu()
        return self.conv2(x, edge_index)
