import torch.nn as nn
import torch
from models.model import Model
from .utils import generate_torch_datasets
from torch.utils.data import DataLoader, Dataset


class TemporalLSTMModel(Model):
    def __init__(self, data, device, batch_size=128) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.model = TemporalLSTM()

        # data needs to be shape TxNxF T temporal, N nodes, F features
        train, test = generate_torch_datasets(data=data, seq_len=12, pre_len=3)
        self.train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.eval_loader = DataLoader(
            test,
            batch_size=self.batch_size,
        )

    def train(self):
        ...

    def predict(self):
        ...


class TemporalLSTM(nn.Module):
    ...
