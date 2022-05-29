from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from torch import embedding

from tasks import Task

# from ..model import Model


@dataclass
class Evaluation:
    models: Dict[str, any] = field(default_factory=dict)
    tasks: Dict[str, any] = field(default_factory=dict)

    def run(self, save_dir=None) -> List[Tuple[str, pd.DataFrame]]:
        res = []
        embs = [(n, m.load_emb()) for n, m in self.models.items()]
        for name, task in self.tasks.items():
            df = pd.DataFrame(columns=list(task.metrics.keys()))
            for n, emb in embs:
                row = pd.Series(task.evaluate(emb=emb))
                row.name = n
                df = df.append(row)
            res.append((name, df))
            if save_dir:
                df.to_csv(save_dir + "/" + name + ".csv")
        return res

    def register_model(self, name, model):
        """
        Register Model for Evaluation.
        Model must inherit from base class Model and must be already trained.
        """
        self.models[name] = model

    def register_task(self, name, task: Task):
        self.tasks[name] = task
