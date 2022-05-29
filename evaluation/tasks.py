from abc import ABC, abstractclassmethod
from typing import Dict

import numpy as np
from sklearn import model_selection
from sklearn.base import clone


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
        for name, (metric, args) in self.metrics.items():
            res[name] = metric(y_test, decoder.predict(X_test), **args)

        return res

    def register_metric(self, name, metric_func, args):
        self.metrics[name] = (metric_func, args)


class TravelTimeEstimation(Task):
    ...
