from ast import literal_eval
from dataclasses import dataclass, field
from typing import List

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import swifter
import torch
import torch_geometric.transforms as T
from shapely.geometry import LineString
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


class Trajectory:
    def __init__(self, df_path: str):
        self.df = pd.read_csv(df_path, sep=";")

    def generate_TTE_datatset(self):
        """
        Generates dataset for TimeTravel Estimation.
        Returns dataframe with traversed road segments and needed time
        """
        tte = self.df[["id", "cpath", "duration"]].copy()
        tte["cpath"] = tte["cpath"].swifter.apply(literal_eval)
        tte["duration"] = tte["duration"].swifter.apply(literal_eval)
        tte["travel_time"] = tte["duration"].swifter.apply(np.sum)
        tte.drop("duration", axis=1, inplace=True)

        return tte.rename(columns={"cpath": "seg_seq"})
