import argparse
import os
import random
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import torch
from models import (
    GAEModel,
    GCNEncoder,
    GTCModel,
    GTNModel,
    Node2VecModel,
    Traj2VecModel,
)
from sklearn.model_selection import train_test_split

from evaluate_models import generate_dataset
from evaluation import Evaluation
from tasks.task_loader import *

model_info = {
    "gtn": {
        "path": "../models/model_states/gtn/model_base_porto_69.pt",
        "dim": 256,
    },
    "gtn_t": {
        "path": "../models/model_states/gtn/ablation/model_base_ablation_None_seed_69.pt",
        "dim": 256,
    },
    "gtn_gcn": {
        "path": "../models/model_states/gtn/ablation/model_base_ablation_gae_seed_69.pt",
        "dim": 256,
    },
    "gtn_dw": {
        "path": "../models/model_states/gtn/ablation/model_base_ablation_dw_seed_69.pt",
        "dim": 256,
    },
}


def init_gtn_variant(name, data, device, network):
    gtn = GTNModel(
        data,
        device,
        network,
        batch_size=512,
        emb_dim=model_info[name]["dim"],
        hidden_dim=512,
    )
    gtn.load_model(model_info[name]["path"])

    return gtn


def ablation(args, data, network, test):
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

        for model_name, _ in model_info.items():
            model = init_gtn_variant(model_name, data, device, network)
            eva.register_model(model_name, model, {})

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

    path = os.path.join(args["path"], "gtn_ablation")
    if not os.path.exists(path):
        os.mkdir(path)
    for name, res in results:
        res.to_csv(path + "/" + name + ".csv")


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
        "-r",
        "--nrows",
        help="Trajectory sample count to load for evaluation",
        type=int,
        required=True,
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

    parser.add_argument(
        "-c",
        "--city",
        help="trajectory dataset to evaluate on",
        type=str,
        default="porto",
    )  # sf, porto or hannover

    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size for lstm training",
        required=False,
        type=int,
        default=512,
    )

    args = vars(parser.parse_args())

    network, test, data = generate_dataset(args, seed=69)
    ablation(args, data, network, test)
