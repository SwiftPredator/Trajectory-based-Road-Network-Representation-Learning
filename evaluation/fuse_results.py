import argparse
import glob
import os

import pandas as pd

TASK_NAMES = ["roadclf", "meanspeed", "timetravel", "nextlocation", "destination"]


def fuse(args):
    root = args["path"]
    for name in TASK_NAMES:
        df = pd.DataFrame()
        for filename in glob.iglob(root + f"**/{name}.csv", recursive=True):
            temp = pd.read_csv(filename)
            df = pd.concat([df, temp], axis=0)
        if df.shape[0] > 0:
            df.to_csv(os.path.join(root, f"{name}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result File Fusion")
    parser.add_argument(
        "-p",
        "--path",
        help="Path to files",
        required=True,
        type=str,
    )

    args = vars(parser.parse_args())

    fuse(args)
