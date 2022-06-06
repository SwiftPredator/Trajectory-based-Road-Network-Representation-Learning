from abc import ABC, abstractclassmethod

import numpy as np


class Task(ABC):
    """
    Base class for all evaluation tasks
    """

    @abstractclassmethod
    def evaluate(emb: np.ndarray):
        ...
