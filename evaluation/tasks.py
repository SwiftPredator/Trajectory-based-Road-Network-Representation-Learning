from abc import ABC, abstractclassmethod
from typing import Dict

import numpy as np
import torch.nn as nn
from sklearn import model_selection
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class Task(ABC):
    """
    Base class for all evaluation tasks
    """

    @abstractclassmethod
    def evaluate(emb: np.ndarray):
        ...


class RoadTypeClfTask(Task):
    def __init__(self, decoder, y):
        self.decoder = decoder
        self.metrics = {}
        self.y = y

    def evaluate(self, emb: np.ndarray) -> Dict[str, any]:
        # create train/test split
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            emb, self.y, test_size=0.2, random_state=1
        )

        # train decoder
        decoder = clone(self.decoder)
        decoder.fit(X_train, y_train)

        # calculate metrics
        res = {}
        for name, (metric, args, proba) in self.metrics.items():
            if proba:
                res[name] = metric(y_test, decoder.predict_proba(X_test), **args)
            else:
                res[name] = metric(y_test, decoder.predict(X_test), **args)

        return res

    def register_metric(self, name, metric_func, args, proba=False):
        self.metrics[name] = (metric_func, args, proba)


class TravelTimeEstimation(Task):
    def __init__(self, traj_dataset, emb_dim=128):
        self.metrics = {}
        self.X = traj_dataset["seg_seq"]
        self.y = traj_dataset["travel_time"]
        self.model = TTE_LSTM(emb_dim=128)

    def evaluate(emb: np.ndarray):
        # make a train test split on trajectorie data

        # train on x trajectories

        # eval on rest
        ...

    def register_metric(self, name, metric_func, args):
        self.metrics[name] = (metric_func, args)


class TTE_LSTM(nn.Module):
    def __init__(self, emb_dim: int = 128):
        self.encoder = nn.LSTM(emb_dim, 256)
        self.decoder = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))

    def forward(self):
        ...

    def train(self):
        ...
