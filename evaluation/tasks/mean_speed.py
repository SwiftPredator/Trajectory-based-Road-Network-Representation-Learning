from typing import Dict

import numpy as np
from sklearn import model_selection
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from .task import Task


class MeanSpeedRegTask(Task):
    def __init__(self, decoder, y, seed):
        self.decoder = decoder
        self.metrics = {}
        self.y = y
        self.seed = seed

    def evaluate(self, emb: np.ndarray) -> Dict[str, any]:

        # calculate metrics
        res = {}
        for name, (metric, args, proba) in self.metrics.items():
            decoder = clone(self.decoder)
            scorer = make_scorer(metric, **args, needs_proba=proba)
            res[name] = np.mean(
                cross_val_score(
                    estimator=decoder, X=emb, y=self.y, scoring=scorer, cv=5
                )
            )

        return res

    def register_metric(self, name, metric_func, args, proba=False):
        self.metrics[name] = (metric_func, args, proba)
