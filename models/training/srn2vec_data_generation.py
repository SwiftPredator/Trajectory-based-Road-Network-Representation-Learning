import os
import sys

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import json

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from generator import RoadNetwork
from models import SRN2VecModel
from torch.utils.data import DataLoader
from tqdm import tqdm

network = RoadNetwork()
network.load("../../osm_data/porto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRN2VecModel(None, device, network)
model.generate_data(
    n_shortest_paths=1280, number_negative=7, window_size=900, save_batch_size=32
) # paras from paper
