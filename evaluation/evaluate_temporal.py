import argparse
import logging
import os
import random
import sys
import uuid
from datetime import datetime

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch_geometric.transforms as T
from generator import RoadNetwork, Trajectory
from models import (
    GAEModel,
    GATEncoder,
    GCNEncoder,
    GTCModel,
    GTNModel,
    Node2VecModel,
    PCAModel,
    TemporalGraphTrainer,
    Traj2VecModel,
)

from evaluation import Evaluation
from tasks import NormalPlugin, TemporalEmbeddingPlugin
from tasks.task_loader import *

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

model_map = {
    "tgtc": (
        TemporalGraphTrainer,
        {"use_attention": False},
        "../models/model_states/temporal/tgtc/model_small_10epochs_lstm_tsd.pt",
    ),
    "tgtc-att": (
        TemporalGraphTrainer,
        {"use_attention": True},
        "../models/model_states/temporal/tgtc/model_small_10epochs_lstm_tsd_lstm_att_current.pt",
    ),
    "tgtc-big": (
        TemporalGraphTrainer,
        {"use_attention": True},
        "../models/model_states/temporal/tgtc/model_small_20epochs_lstm_tsd_with_big_att.pt",
    ),
    "traj2vec": (Traj2VecModel, {}, "../models/model_states/traj2vec"),
    "gaegcn": (
        GAEModel,
        {"encoder": GCNEncoder},
        "../models/model_states/temporal/gaegcn/model.pt",
    ),
    "gaegat": (
        GAEModel,
        {"encoder": GATEncoder},
        "../models/model_states/gaegat/",
    ),
    "node2vec": (Node2VecModel, {"q": 4, "p": 1}, "../models/model_states/node2vec"),
    "deepwalk": (Node2VecModel, {"q": 1, "p": 1}, "../models/model_states/deepwalk"),
    "pca": (PCAModel, {}, "../models/model_states/pca"),
}


def load_data():
    logging.info("Load data...")
    unmapped_traj = pd.read_csv(
        "../datasets/trajectories/hanover/temporal/mapped_id_poly_clipped_small_graph.csv",
        ";",
    )
    traj = Trajectory(
        "../datasets/trajectories/hanover/temporal/road_segment_map_final_small_graph.csv",
        nrows=100000000,
    ).generate_TTE_datatset()
    traj["seg_seq"] = traj["seg_seq"].map(np.array)
    traj = traj.join(
        unmapped_traj[["start_stamp", "end_stamp", "id"]].set_index("id"),
        on="id",
        how="left",
    )
    traj["start_stamp"] = pd.to_datetime(traj["start_stamp"], unit="s")
    traj["end_stamp"] = pd.to_datetime(traj["end_stamp"], unit="s")

    traj["dayofweek"] = traj["start_stamp"].dt.dayofweek
    traj["start_hour"] = traj["start_stamp"].dt.hour
    traj["end_hour"] = traj["end_stamp"].dt.hour

    network = RoadNetwork()
    network.load_edges("../osm_data/hanover_temp_small_graph/")
    network.gdf_edges.rename(
        columns={"speed_limi": "speed_limit", "highway_en": "highway_enc"}, inplace=True
    )

    data = network.generate_road_segment_pyg_dataset(
        traj_data=None, include_coords=False, dataset="hannover_small"
    )
    logging.info("...finished loading data")

    return network, traj, data


def load_tsd_embedding(network, device):
    logging.info("Load TSD Embedding...")
    data = network.generate_road_segment_pyg_dataset(only_edge_index=True)
    tsd = Traj2VecModel(data, network, device=device)
    tsd.load_model("../models/training/temporal/models/model_tsd_small.pt")
    tsd_emb = tsd.load_emb()
    logging.info("...finished loading TSD")

    return tsd_emb


def load_temporal_plugin(model, network, device):
    plugin = TemporalEmbeddingPlugin(model, network, device)
    plugin.load_data("../models/training/temporal/plugin_data_small.pt")

    return plugin


def evaluate_models(args, data, traj, network, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(args["device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose(
        [
            T.ToDevice(device),
        ]
    )
    data = transform(data)
    models = [m for m in args["models"].split(",")]
    tasks = [t for t in args["tasks"].split(",")]
    eva = Evaluation()

    if "roadclf" in tasks:
        eva.register_task("roadclf", init_roadclf(args, network, seed))

    if "traveltime" in tasks:
        eva.register_task(
            "timetravel", init_traveltime(args, traj, network, device, seed)
        )

    if "meanspeed" in tasks:
        eva.register_task("meanspeed", init_meanspeed(args, network, seed))

    if "nextlocation" in tasks:
        eva.register_task(
            "nextlocation", init_nextlocation(args, traj, network, device, seed)
        )

    if "destination" in tasks:
        eva.register_task(
            "destination", init_destination(args, traj, network, device, seed)
        )

    for m in models:
        model, margs, model_path = model_map[m]
        mdata = data
        if "tgtc" in m:
            margs["adj"] = np.loadtxt(
                f"../models/training/temporal/traj_adj_k_2_bi_temporal_gtc_small_graph.gz"
            )
            margs["struc_emb"] = load_tsd_embedding(network, device)

            mdata = torch.load(
                "../datasets/trajectories/hanover/temporal/temporal_data_small.pt"
            )
            mdata = torch.swapaxes(mdata, 0, 1)
            mdata = mdata.numpy()

            margs["device_ids"] = [int(args["device"])]

        model = model(mdata, device=device, **margs)
        model.load_model(path=model_path)

        targs = {}
        if "roadclf" not in tasks:
            if "tgtc" in m:
                print("adding temporal plugin")
                plugin = load_temporal_plugin(model.model, network, device)
                targs = {"plugin": plugin}
            else:
                plugin = NormalPlugin(model, network, device)
                targs = {"plugin": plugin}

        eva.register_model(m, model, targs)

    res = eva.run()

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Evaluation")
    parser.add_argument(
        "-m", "--models", help="Models to evaluate", required=True, type=str
    )
    parser.add_argument(
        "-t", "--tasks", help="Tasks to evaluate on", required=True, type=str
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Epochs to train lstm for (trajectory task evaluation)",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size for lstm training",
        required=False,
        type=int,
        default=32,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Save path evaluation results",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-d", "--device", help="Cuda device number", type=int, required=True
    )

    args = vars(parser.parse_args())

    seed = 69
    results = []

    network, traj, data = load_data()
    res = evaluate_models(args, data, traj, network, seed)

    for name, res in res:
        res["seed"] = seed
        results.append((name, res))

    path = os.path.join(
        args["path"],
        str(datetime.now().strftime("%m-%d-%Y-%H-%M")),
    )
    os.mkdir(path)
    for name, res in results:
        res.to_csv(path + "/" + name + ".csv")
