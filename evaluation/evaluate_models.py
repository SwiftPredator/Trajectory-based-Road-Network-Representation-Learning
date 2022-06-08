import argparse
import os
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
from models import GAEModel, GATEncoder, GCNEncoder, Node2VecModel, PCAModel, Toast
from sklearn import linear_model, metrics

from evaluation import Evaluation
from tasks import MeanSpeedRegTask, RoadTypeClfTask, TravelTimeEstimation

model_map = {
    "gaegcn": (GAEModel, {"encoder": GCNEncoder}),
    "gaegat": (GAEModel, {"encoder": GATEncoder}),
    "node2vec": (Node2VecModel, {"q": 4, "p": 1}),
    "deepwalk": (Node2VecModel, {"q": 1, "p": 1}),
    "pca": (PCAModel, {}),
    "toast": (Toast, {}),
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

    if args["speed"]:
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

        return (
            network,
            traj_dataset,
            network.generate_road_segment_pyg_dataset(traj_data=traj_features),
        )
    else:
        return network, traj_dataset, network.generate_road_segment_pyg_dataset()


# index is correct
def init_roadclf(network):
    decoder = linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000)
    y = np.array(
        [network.gdf_edges.loc[n]["highway_enc"] for n in network.line_graph.nodes]
    )
    roadclf = RoadTypeClfTask(decoder, y)
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
def init_meanspeed(network):
    tf = pd.read_csv("../datasets/trajectories/Porto/speed_features_unnormalized.csv")
    tf.set_index(["u", "v", "key"], inplace=True)
    map_id = {j: i for i, j in enumerate(network.line_graph.nodes)}
    tf["idx"] = tf.index.map(map_id)
    tf.sort_values(by="idx", axis=0, inplace=True)
    decoder = linear_model.LinearRegression(fit_intercept=True)
    y = tf["avg_speed"]
    y.fillna(0, inplace=True)
    y = y.round(2)
    mean_speed_reg = MeanSpeedRegTask(decoder, y)

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


def evaluate_model(args, data, network, trajectory):
    """
    trains the model given by the args argument on the corresponding data

    Args:
        args (dict): Args from arg parser
        data (pyg Data): Data to train on
    """
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
        eva.register_task("roadclf", init_roadclf(network))

    if "traveltime" in tasks:
        eva.register_task(
            "timetravel", init_traveltime(args, trajectory, network, device)
        )

    if "meanspeed" in tasks:
        eva.register_task("meanspeed", init_meanspeed(network))

    for m in models:
        model, margs = model_map[m]
        model = model(data, device=device, **margs)
        model.load_model(path=os.path.join("../models/model_states", m, "model.pt"))
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
        help="Include speed features (true or false)",
        type=bool,
        default=False,
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

    args = vars(parser.parse_args())

    network, trajectory, data = generate_dataset(args)
    evaluate_model(args, data, network, trajectory)
