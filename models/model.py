from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all evaluated models
    """

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def load_model(self):
        ...

    @abstractmethod
    def load_emb(self):
        ...
