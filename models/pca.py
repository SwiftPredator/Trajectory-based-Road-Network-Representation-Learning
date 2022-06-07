import numpy as np
from sklearn.decomposition import PCA

from .model import Model


class PCAModel(Model):
    """
    Principal component analysis wrapper over sklearns implementation.
    """

    def __init__(self, data, device=None, emb_dim=4):
        self.pca = PCA(n_components=emb_dim)
        self.data = data

    def train(self):
        self.emb = self.pca.fit_transform(self.data.x.detach().cpu().numpy())
        self.save_emb("")

    def save_emb(self, path):
        np.savetxt(
            path + "embedding.out",
            X=self.emb,
        )

    def load_model(self, path):  # placeholder
        ...

    def load_emb(self, path=None):
        if path:
            return np.loadtxt(path)
        return self.pca.fit_transform(self.data.x.detach().cpu().numpy())
