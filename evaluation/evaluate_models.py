import argparse
import os
import random
import sys
import uuid
from datetime import datetime

import pandas as pd
import torch

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
    HRNRModel,
    Node2VecModel,
    PCAModel,
    RFNModel,
    SRN2VecModel,
    Toast,
    Traj2VecModel,
)
from sklearn import linear_model, metrics

from evaluation import Evaluation
from tasks import (
    DestinationPrediciton,
    MeanSpeedRegTask,
    NextLocationPrediciton,
    RoadTypeClfTask,
    RoutePlanning,
    TravelTimeEstimation,
)

model_map = {
    "gtn": (GTNModel, {}, "../models/model_states/gtn/"),
    "gtc": (GTCModel, {}, "../models/model_states/gtc/"),
    "traj2vec": (Traj2VecModel, {}, "..models/model_states/traj2vec"),
    "gaegcn": (
        GAEModel,
        {"encoder": GCNEncoder},
        "../models/model_states/gaegcn/",
    ),
    "gaegat": (
        GAEModel,
        {"encoder": GATEncoder},
        "../models/model_states/gaegat/",
    ),
    "node2vec": (Node2VecModel, {"q": 4, "p": 1}, "../models/model_states/node2vec"),
    "deepwalk": (Node2VecModel, {"q": 1, "p": 1}, "../models/model_states/deepwalk"),
    "pca": (PCAModel, {}, "../models/model_states/pca"),
    "toast": (Toast, {}, "../models/model_states/toast"),
    "srn2vec": (SRN2VecModel, {}, "../models/model_states/srn2vec"),
    "rfn": (RFNModel, {}, "../models/model_states/rfn"),
    "hrnr": (
        HRNRModel,
        {"data_path": "../models/training/hrnr_data"},
        "../models/model_states/hrnr",
    ),
}


def generate_dataset(args):
    """
    Generates the dataset with optional speed features.

    Args:
        args (dict): args from argparser

    Returns:
        pyg Data: Dataset as pyg Data Object with x and edge_index
    """
    network = RoadNetwork()
    network.load("../osm_data/porto")
    trajectory = Trajectory(
        "../datasets/trajectories/Porto/road_segment_map_final.csv", nrows=args["nrows"]
    )
    traj_dataset = trajectory.generate_TTE_datatset()

    if args["speed"] == 1:
        traj_features = pd.read_csv(
            "../datasets/trajectories/Porto/speed_features_unnormalized.csv"
        )
        traj_features.set_index(["u", "v", "key"], inplace=True)
        traj_features["util"] = (
            traj_features["util"] - traj_features["util"].min()
        ) / (
            traj_features["util"].max() - traj_features["util"].min()
        )  # min max normalization
        traj_features["avg_speed"] = (
            traj_features["avg_speed"] - traj_features["avg_speed"].min()
        ) / (
            traj_features["avg_speed"].max() - traj_features["avg_speed"].min()
        )  # min max normalization
        traj_features.fillna(0, inplace=True)

        drop_label = [args["drop_label"]] if args["drop_label"] is not None else []
        return (
            network,
            traj_dataset,
            network.generate_road_segment_pyg_dataset(
                traj_data=traj_features,
                include_coords=True,
                drop_labels=drop_label,
            ),
        )
    else:
        print("without speed")
        return (
            network,
            traj_dataset,
            network.generate_road_segment_pyg_dataset(
                include_coords=True, drop_labels=drop_label
            ),
        )


# index is correct
def init_roadclf(args, network):
    decoder = linear_model.LogisticRegression(
        multi_class="multinomial", max_iter=1000, n_jobs=-1
    )
    y = np.array(
        [network.gdf_edges.loc[n]["highway_enc"] for n in network.line_graph.nodes]
    )
    roadclf = RoadTypeClfTask(decoder, y, seed=args["seed"])
    roadclf.register_metric(
        name="f1_micro", metric_func=metrics.f1_score, args={"average": "micro"}
    )
    roadclf.register_metric(
        name="f1_macro", metric_func=metrics.f1_score, args={"average": "macro"}
    )
    roadclf.register_metric(
        name="f1_weighted",
        metric_func=metrics.f1_score,
        args={"average": "weighted"},
    )
    roadclf.register_metric(
        name="accuracy",
        metric_func=metrics.accuracy_score,
        args={"normalize": True},
    )
    roadclf.register_metric(
        name="AUC",
        metric_func=metrics.roc_auc_score,
        args={"multi_class": "ovo"},
        proba=True,
    )

    return roadclf


def init_traveltime(args, traj_data, network, device):
    travel_time_est = TravelTimeEstimation(
        traj_dataset=traj_data,
        network=network,
        device=device,
        batch_size=128,
        epochs=args["epochs"],
        seed=args["seed"],
    )
    travel_time_est.register_metric(
        name="MSE", metric_func=metrics.mean_squared_error, args={}
    )
    travel_time_est.register_metric(
        name="MAE", metric_func=metrics.mean_absolute_error, args={}
    )
    travel_time_est.register_metric(
        name="RMSE", metric_func=metrics.mean_squared_error, args={"squared": False}
    )
    travel_time_est.register_metric(
        name="MAPE", metric_func=metrics.mean_absolute_percentage_error, args={}
    )

    return travel_time_est


# label index is right here;
def init_meanspeed(args, network):
    tf = pd.read_csv("../datasets/trajectories/Porto/speed_features_unnormalized.csv")
    tf.set_index(["u", "v", "key"], inplace=True)
    map_id = {j: i for i, j in enumerate(network.line_graph.nodes)}
    tf["idx"] = tf.index.map(map_id)
    tf.sort_values(by="idx", axis=0, inplace=True)
    decoder = linear_model.LinearRegression(fit_intercept=True)
    y = tf["avg_speed"]
    y.fillna(0, inplace=True)
    y = y.round(2)
    mean_speed_reg = MeanSpeedRegTask(decoder, y, seed=args["seed"])

    mean_speed_reg.register_metric(
        name="MSE", metric_func=metrics.mean_squared_error, args={}
    )
    mean_speed_reg.register_metric(
        name="MAE", metric_func=metrics.mean_absolute_error, args={}
    )
    mean_speed_reg.register_metric(
        name="RMSE", metric_func=metrics.mean_squared_error, args={"squared": False}
    )

    return mean_speed_reg


def init_nextlocation(args, traj_data, network, device):
    nextlocation_pred = NextLocationPrediciton(
        traj_dataset=traj_data,
        network=network,
        device=device,
        batch_size=256,
        epochs=args["epochs"],
        seed=args["seed"],
    )

    nextlocation_pred.register_metric(
        name="accuracy",
        metric_func=metrics.accuracy_score,
        args={"normalize": True},
    )

    return nextlocation_pred


def init_destination(args, traj_data, network, device):
    destination_pred = DestinationPrediciton(
        traj_dataset=traj_data,
        network=network,
        device=device,
        batch_size=256,
        epochs=args["epochs"],
        seed=args["seed"],
    )

    destination_pred.register_metric(
        name="accuracy",
        metric_func=metrics.accuracy_score,
        args={"normalize": True},
    )

    return destination_pred


def init_route(args, traj_data, network, device):
    route_pred = RoutePlanning(
        traj_dataset=traj_data,
        network=network,
        device=device,
        batch_size=256,
        epochs=args["epochs"],
        seed=args["seed"],
    )

    route_pred.register_metric(
        name="accuracy",
        metric_func=metrics.accuracy_score,
        args={"normalize": True},
    )
    route_pred.register_metric(
        name="f1_micro", metric_func=metrics.f1_score, args={"average": "micro"}
    )
    route_pred.register_metric(
        name="f1_macro", metric_func=metrics.f1_score, args={"average": "macro"}
    )
    route_pred.register_metric(
        name="f1_weighted",
        metric_func=metrics.f1_score,
        args={"average": "weighted"},
    )

    return route_pred


def evaluate_model(args, data, network, trajectory):
    """
    trains the model given by the args argument on the corresponding data

    Args:
        args (dict): Args from arg parser
        data (pyg Data): Data to train on
    """
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
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
        eva.register_task("roadclf", init_roadclf(args, network))

    if "traveltime" in tasks:
        eva.register_task(
            "timetravel", init_traveltime(args, trajectory, network, device)
        )

    if "meanspeed" in tasks:
        eva.register_task("meanspeed", init_meanspeed(args, network))

    if "nextlocation" in tasks:
        eva.register_task(
            "nextlocation", init_nextlocation(args, trajectory, network, device)
        )

    if "destination" in tasks:
        eva.register_task(
            "destination", init_destination(args, trajectory, network, device)
        )

    if "route" in tasks:
        eva.register_task("route", init_route(args, trajectory, network, device))

    for m in models:
        model, margs, model_path = model_map[m]
        model_file_name = "model.params" if m == "rfn" else "model.pt"
        if m in ["toast", "srn2vec", "rfn", "hrnr"]:
            margs["network"] = network
        if m in ["gaegcn_no_features", "gaegat_no_features"]:
            data.x = None
            data = T.OneHotDegree(128)(data)
        if m in ["gtn_no_speed", "gtn_speed"]:
            margs["network"] = network
            margs["traj_data"] = trajectory

        model = model(data, device=device, **margs)
        model.load_model(path=os.path.join(model_path, model_file_name))
        eva.register_model(m, model)

    path = os.path.join(
        args["path"],
        str(datetime.now().strftime("%m-%d-%Y-%H-%M")),
    )
    os.mkdir(path)

    eva.run(save_dir=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Evaluation")
    parser.add_argument(
        "-m", "--models", help="Models to evaluate", required=True, type=str
    )
    parser.add_argument(
        "-t", "--tasks", help="Tasks to evaluate on", required=True, type=str
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Epochs to train lstm for (time travel evaluation)",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--speed",
        help="Include speed features (1 or 0)",
        type=int,
        default=0,
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
        "-r",
        "--nrows",
        help="Trajectory sample count to use for evaluation",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-se",
        "--seed",
        help="Seed for the random operations like train/test split",
        default=69,
        type=int,
    )

    parser.add_argument(
        "-dl",
        "--drop_label",
        help="remove label from train dataset",
        type=str,
    )

    args = vars(parser.parse_args())

    network, trajectory, data = generate_dataset(args)
    evaluate_model(args, data, network, trajectory)
