import argparse
import os
import sys
import uuid
from datetime import datetime

import pandas as pd
import torch

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch_geometric.transforms as T
from generator import RoadNetwork
from models import (GAEModel, GATEncoder, GCNEncoder, Node2VecModel, PCAModel,
                    Toast)

model_map = {
    "gaegcn": (GAEModel, {"encoder": GCNEncoder}),
    "gaegat": (GAEModel, {"encoder": GATEncoder}),
    "node2vec": (Node2VecModel, {"q": 4, "p": 1}),
    "deepwalk": (Node2VecModel, {"q": 1, "p": 1}),
    "pca": (PCAModel, {}),
    "toast": (Toast, {}),  # needs to be fixed
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
    network.load("../../osm_data/porto")

    if args["speed"]:
        traj_features = pd.read_csv(
            "../../datasets/trajectories/Porto/speed_features_unnormalized.csv"
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

        return network, network.generate_road_segment_pyg_dataset(
            traj_data=traj_features
        )
    else:
        return network, network.generate_road_segment_pyg_dataset()


def train_model(args, data, network):
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
    model, margs = model_map[args["model"]]
    if args["model"] == "toast":
        margs["network"] = network
    model = model(data, device=device, emb_dim=args["embedding"], **margs)
    model.train(epochs=args["epochs"])
    path = os.path.join(
        args["path"],
        args["model"]
        + "_"
        + str(args["epochs"])
        + "_"
        + str(datetime.now().strftime("%m-%d-%Y-%H-%M")),
    )
    os.mkdir(path)
    model.save_model(path=path)
    model.save_emb(path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Training")
    parser.add_argument("-m", "--model", help="Model to train", required=True, type=str)
    parser.add_argument(
        "-e", "--epochs", help="Epochs to train for", required=True, type=int
    )
    parser.add_argument(
        "-s",
        "--speed",
        help="Include speed features (true or false)",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-emb", "--embedding", help="Embedding Dimension", type=int, required=True
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Save path for model and embedding",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-d", "--device", help="Cuda device number", type=int, required=True
    )

    args = vars(parser.parse_args())

    network, data = generate_dataset(args)
    train_model(args, data, network)
