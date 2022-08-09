import argparse
import os
import random
import sys
from turtle import speed

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import torch
from generator import RoadNetwork, Trajectory
from models import GTCModel
from sklearn.model_selection import train_test_split

from evaluate_models import generate_dataset
from evaluation import Evaluation
from tasks.task_loader import *

model_info = {
    "gtc_base": {
        "path": "../models/model_states/gtc/feature_analysis/model_base_normal.pt",
    },
    "gtc_base_traj_all": {
        "path": "../models/model_states/gtc/feature_analysis/model_base_traj_all.pt",
    },
    "gtc_base_traj_speed": {
        "path": "../models/model_states/gtc/feature_analysis/model_base_traj_speed.pt",
    },
    "gtc_base_traj_util": {
        "path": "../models/model_states/gtc/feature_analysis/model_base_traj_util.pt",
    },
    "gtc_roadclf": {
        "path": "../models/model_states/gtc/feature_analysis/model_roadclf_normal.pt",
    },
    "gtc_roadclf_traj_all": {
        "path": "../models/model_states/gtc/feature_analysis/model_roadclf_traj_all.pt",
    },
    "gtc_roadclf_traj_speed": {
        "path": "../models/model_states/gtc/feature_analysis/model_roadclf_traj_speed.pt",
    },
    "gtc_roadclf_traj_util": {
        "path": "../models/model_states/gtc/feature_analysis/model_roadclf_traj_util.pt",
    },
}


def init_gtc_variant(name, data, device, network, adj):
    gtc = GTCModel(data, device, network, adj=adj)
    gtc.load_model(model_info[name]["path"])

    return gtc


def ablation(args, data_normal, data_util, data_speed, data_all, network, test):
    seeds = [69]
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args["device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for seed in seeds:
        print(f"START Evaluation with seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        args["seed"] = seed
        args["epochs"] = 10
        tasks = [t for t in args["tasks"].split(",")]
        adj = np.loadtxt("../models/training/gtn_precalc_adj/traj_adj_k_2.gz")
        eva = Evaluation()

        if "roadclf" in tasks:
            eva.register_task("roadclf", init_roadclf(args, network, seed=seed))
        if "meanspeed" in tasks:
            eva.register_task("meanspeed", init_meanspeed(args, network, seed=seed))
        if "traveltime" in tasks:
            eva.register_task(
                "traveltime",
                init_traveltime(args, test, network, device, seed=seed),
            )
        if "nextlocation" in tasks:
            eva.register_task(
                "nextlocation",
                init_nextlocation(args, test, network, device, seed=seed),
            )
        if "destination" in tasks:
            eva.register_task(
                "destination",
                init_destination(args, test, network, device, seed=seed),
            )

        # load only roadclf if roadclf is the task
        type = "roadclf" if "roadclf" in tasks else "base"
        models = {k: model_info[k] for k in model_info.keys() if type in k}
        for model_name, _ in models.items():
            if "util" in model_name:
                model = init_gtc_variant(model_name, data_util, device, network, adj)
            elif "speed" in model_name:
                model = init_gtc_variant(model_name, data_speed, device, network, adj)
            elif "all" in model_name:
                model = init_gtc_variant(model_name, data_all, device, network, adj)
            else:
                model = init_gtc_variant(model_name, data_normal, device, network, adj)

            eva.register_model(model_name, model)

        seed_results = eva.run()

        # concatinate results
        if len(results) == 0:
            for name, res in seed_results:
                res["seed"] = seed
                results.append((name, res))
        else:
            for i, ((name, full_res), (_, seed_res)) in enumerate(
                zip(results, seed_results)
            ):
                seed_res["seed"] = seed
                results[i] = (name, pd.concat([full_res, seed_res], axis=0))

    path = os.path.join(args["path"], "gtc_feature_eval")
    if not os.path.exists(path):
        os.mkdir(path)
    for name, res in results:
        res.to_csv(path + "/" + name + ".csv")


def generate_eval_datasets():
    network = RoadNetwork()
    network.load("../osm_data/porto")
    drop_label = [args["drop_label"]] if args["drop_label"] is not None else []
    traj_features = pd.read_csv(
        "../datasets/trajectories/Porto/speed_features_unnormalized.csv"
    )

    traj_features.set_index(["u", "v", "key"], inplace=True)
    traj_features["util"] = (traj_features["util"] - traj_features["util"].min()) / (
        traj_features["util"].max() - traj_features["util"].min()
    )  # min max normalization
    traj_features["avg_speed"] = (
        traj_features["avg_speed"] - traj_features["avg_speed"].min()
    ) / (
        traj_features["avg_speed"].max() - traj_features["avg_speed"].min()
    )  # min max normalization
    traj_features.fillna(0, inplace=True)

    return (
        network.generate_road_segment_pyg_dataset(
            traj_data=None,
            include_coords=True,
            drop_labels=drop_label,
        ),
        network.generate_road_segment_pyg_dataset(
            traj_data=traj_features[["id", "util"]].copy(),
            include_coords=True,
            drop_labels=drop_label,
        ),
        network.generate_road_segment_pyg_dataset(
            traj_data=traj_features[["id", "avg_speed"]].copy(),
            include_coords=True,
            drop_labels=drop_label,
        ),
        network.generate_road_segment_pyg_dataset(
            traj_data=traj_features[["id", "util", "avg_speed"]].copy(),
            include_coords=True,
            drop_labels=drop_label,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Evaluation")
    parser.add_argument(
        "-t", "--tasks", help="Tasks to evaluate on", required=True, type=str
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

    parser.add_argument(
        "-dl",
        "--drop_label",
        help="remove label from train dataset",
        type=str,
    )

    args = vars(parser.parse_args())
    network, traj_test, _ = generate_dataset(args, 69)  # default seed
    data_normal, data_util, data_speed, data_all = generate_eval_datasets()
    ablation(args, data_normal, data_util, data_speed, data_all, network, traj_test)
