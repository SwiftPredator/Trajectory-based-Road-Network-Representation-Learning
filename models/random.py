import numpy as np
from sklearn.decomposition import PCA

from .model import Model


class RandomModel(Model):
    """
    Principal component analysis wrapper over sklearns implementation.
    """

    def __init__(self, data, device=None, emb_dim=128):
        self.emb_dim = emb_dim
        self.data = data

    def train(self):
        ...

    def save_emb(self, path):
        ...

    def load_model(self, path):  # placeholder
        ...

    def load_emb(self, path=None):
        return np.random.rand(self.data.x.shape[0], self.emb_dim)
