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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

network = RoadNetwork()
network.load("../../osm_data/porto")
trajectory = Trajectory(
    "../../datasets/trajectories/Porto/road_segment_map_final.csv", nrows=10000000
).generate_TTE_datatset()
traj_features = pd.read_csv(
    "../../datasets/trajectories/Porto/speed_features_unnormalized.csv"
)
traj_features.set_index(["u", "v", "key"], inplace=True)
traj_features.fillna(0, inplace=True)

# data_roadclf = network.generate_road_segment_pyg_dataset(
#     include_coords=True, drop_labels=["highway_enc"], traj_data=None
# )
data_rest = network.generate_road_segment_pyg_dataset(
    include_coords=True, traj_data=None
)

adj = np.loadtxt("./gtn_precalc_adj/traj_adj_k_1.gz")
adj_sample = np.loadtxt("./gtn_precalc_adj/traj_adj_k_1_False_no_selfloops_smoothed.gz")

# create init emb from gtc and traj2vec concat
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
traj2vec = Traj2VecModel(data_rest, network, adj_sample, device=device, emb_dim=128)
traj2vec.load_model("../model_states/traj2vec/model_base.pt")
gtc = GTCModel(data_rest, device, network, None, adj=adj)
gtc.load_model("../model_states/gtc/model_base.pt")

init_emb = torch.Tensor(np.concatenate([gtc.load_emb(), traj2vec.load_emb()], axis=1))

train, _ = train_test_split(trajectory, test_size=0.3, random_state=69)

# init GTN Model
model = GTNModel(
    data_rest,
    device,
    network,
    train,
    traj_features,
    init_emb,
    adj_sample,
    batch_size=32,
)

model.train(epochs=25)

torch.save(
    model.model.state_dict(), os.path.join("../model_states/gtn/" + "/model_base.pt")
)
