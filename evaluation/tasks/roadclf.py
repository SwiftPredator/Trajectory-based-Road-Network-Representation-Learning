from typing import Dict

import numpy as np
from sklearn import model_selection
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from .task import Task


class RoadTypeClfTask(Task):
    def __init__(self, decoder, y):
        self.decoder = decoder
        self.metrics = {}
        self.y = y

    def evaluate(self, emb: np.ndarray) -> Dict[str, any]:
        decoder = clone(self.decoder)

        # calculate metrics
        res = {}
        for name, (metric, args, proba) in self.metrics.items():
            scorer = make_scorer(metric, **args, needs_proba=proba)
            res[name] = cross_val_score(
                estimator=decoder, X=emb, y=self.y, scoring=scorer, cv=5
            )

        return res

    def register_metric(self, name, metric_func, args, proba=False):
        self.metrics[name] = (metric_func, args, proba)
