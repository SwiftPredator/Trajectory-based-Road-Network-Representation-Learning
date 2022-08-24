import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec

from .model import Model


class Node2VecModel(Model):
    def __init__(
        self,
        data,
        device,
        emb_dim=128,
        walk_length=30,
        walks_per_node=25,
        context_size=5,
        negative_samples=7,
        q=1,
        p=1,
    ):
        self.model = Node2Vec(
            data.edge_index,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=negative_samples,
            p=p,
            q=q,
            sparse=True,
        ).to(device)
        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=4)
        self.device = device
        self.data = data
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

    def train(self, epochs):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for pos_rw, neg_rw in self.loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss += total_loss / len(self.loader)
            if e > 0 and e % 20 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), path + "model.pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        np.savetxt(path + "embedding.out", X=self.model().detach().cpu().numpy())

    def load_emb(self, path=None):
        if path:
            self.emb = np.loadtxt(path)
        return self.model().detach().cpu().numpy()
