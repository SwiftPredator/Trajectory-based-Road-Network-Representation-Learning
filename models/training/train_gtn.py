import os
import sys

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from evaluation.tasks import NextLocationPrediciton, TravelTimeEstimation
from generator import RoadNetwork, Trajectory
from models import GTCModel, GTNModel, Traj2VecModel
from torch.utils.data import DataLoader
from tqdm import tqdm

network = RoadNetwork()
network.load("../../osm_data/porto")
trajectory = Trajectory(
    "../../datasets/trajectories/Porto/road_segment_map_final.csv", nrows=10000000
).generate_TTE_datatset()

data_roadclf = network.generate_road_segment_pyg_dataset(
    include_coords=True, drop_labels=["highway_enc"], traj_data=None
)
data_rest = network.generate_road_segment_pyg_dataset(
    include_coords=True, traj_data=None
)

adj = np.loadtxt("./gtn_precalc_adj/traj_adj_k_1.gz")

# create init emb from gtc and traj2vec concat
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
traj2vec = Traj2VecModel(data_roadclf, network, adj, device=device, emb_dim=128)
traj2vec.load_model("../model_states/traj2vec/model_base.pt")
gtc = GTCModel(data_roadclf, device, network, None, adj=adj)
gtc.load_model("../model_states/gtc/model_noroad.pt")

init_emb = torch.Tensor(np.concatenate([gtc.load_emb(), traj2vec.load_emb()], axis=1))


# init GTN Model
model = GTNModel(data_roadclf, device, network, trajectory, init_emb, batch_size=256)

model.train(epochs=25)

torch.save(model.model.state_dict(), os.path.join("../model_states/gtn/" + "/model.pt"))
