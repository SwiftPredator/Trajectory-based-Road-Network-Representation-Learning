"""
Generate scores for GTc with different convolutions
Split with different seeds to get better approximation
"""

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
from models import GTNModel
from torch_geometric.nn import (
    APPNP,
    ARMAConv,
    FAConv,
    GatedGraphConv,
    GCN2Conv,
    GCNConv,
    LEConv,
    LGConv,
    SGConv,
    TAGConv,
)

from evaluate_models import (
    generate_dataset,
    init_destination,
    init_meanspeed,
    init_nextlocation,
    init_roadclf,
    init_traveltime,
)
from evaluation import Evaluation


def evaluate_k(args, data, network, trajectory):
    seeds = [42, 69, 88, 420, 1234, 99, 102, 73, 10, 28]
    convs = [
        ("APPNP", APPNP, {"K": 10, "alpha": 0.1, "dropout": 0.5, "cached": True}),
        ("ARMAConv", ARMAConv, {"dropout": 0.5}),
        ("FAConv", FAConv, {"dropout": 0.5, "cached": True}),
        ("GatedGraphConv", GatedGraphConv, {"num_layers": 1}),
        ("GCN2Conv", GCN2Conv, {"alpha": 0.1, "cached": True}),
        ("GCNConv", GCNConv, {"cached": True}),
        ("GCNConv_imp", GCNConv, {"cached": True, "improved": True})(GCNConv, {}),
        ("GCNConv_noself", GCNConv, {"add_self_loops": False})(LEConv, {}),
        ("LGConv", LGConv, {}),
        ("SGConv", SGConv, {"cached": True}),
        ("SGConv_noself", SGConv, {"cached": True, "add_self_loops": False})(
            TAGConv, {"K": 1}
        ),
    ]

    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args["device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        args["seed"] = seed
        args["epochs"] = 10

        eva = Evaluation()
        eva.register_task("roadclf", init_roadclf(args, network))
        eva.register_task("meanspeed", init_meanspeed(args, network))
        eva.register_task(
            "traveltime", init_traveltime(args, trajectory, network, device)
        )
        eva.register_task(
            "nextlocation", init_traveltime(args, trajectory, network, device)
        )
        eva.register_task(
            "destination", init_traveltime(args, trajectory, network, device)
        )

        for name, conv, cargs in convs:
            model = GTNModel(
                data,
                device,
                network,
                trajectory,
                load_traj_adj_path="../models/training/gtn_precalc_adj/traj_adj_k_1.gz",
            )
            model.train(epochs=5000)
            eva.register_model("gtn_k_{}".format(k), model)

        path = os.path.join(args["path"], f"gtn_k_eva_Forward_only_seed_{seed}")
        os.mkdir(path)

        eva.run(save_dir=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Evaluation")
    parser.add_argument(
        "-t", "--tasks", help="Tasks to evaluate on", required=True, type=str
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

    args = vars(parser.parse_args())

    network, trajectory, data = generate_dataset(args)
    evaluate_k(args, data, network, trajectory)
