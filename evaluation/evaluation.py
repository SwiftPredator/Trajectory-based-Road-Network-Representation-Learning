from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from torch import embedding
from tqdm import tqdm

from tasks import Task

# from ..model import Model


@dataclass
class Evaluation:
    models: Dict[str, any] = field(default_factory=dict)
    tasks: Dict[str, any] = field(default_factory=dict)

    def run(self, save_dir=None) -> List[Tuple[str, pd.DataFrame]]:
        res = []
        embs = [(n, m.load_emb(), a) for n, (m, a) in self.models.items()]
        for name, task in tqdm(self.tasks.items(), desc="Current task"):
            df = pd.DataFrame(columns=list(task.metrics.keys()))
            for n, emb, args in tqdm(embs, desc="Current model", leave=False):
                if "gtn" in n:
                    emb = emb[2:]  # remove mask and pad token
                row = pd.Series(task.evaluate(emb=emb, **args))
                row.name = n
                df = df.append(row)
            res.append((name, df))
            if save_dir:
                df.to_csv(save_dir + "/" + name + ".csv")
        return res

    def train(self, margs: Dict, save_parent_dir="../model_states/"):
        for n, (m, args) in (self.models, margs).items():
            m.train(**args)
            m.save_model(save_parent_dir + n + "/")
            m.save_emb(save_parent_dir + n + "/")

    def register_model(self, name, model, args):
        """
        Register Model for Evaluation.
        Model must inherit from base class Model and must be already trained.
        """
        self.models[name] = (model, args)

    def register_task(self, name, task: Task):
        self.tasks[name] = task
