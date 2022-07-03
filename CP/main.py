import os
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from model import CP


def get_args():
    parser = argparse.ArgumentParser(description="")
    ### mode ###
    parser.add_argument(
        "-t",
        "--task",
        default="",
        choices=["custom"],
    )

    ### path ###
    parser.add_argument("--dict", type=str, default="../../dict/CP.pkl")
    parser.add_argument("--input_dir", type=str, default="")

    ### parameter ###
    parser.add_argument("--max_len", type=int, default=512)

    ### output ###
    parser.add_argument("--output_dir", default="../../data/CP")
    parser.add_argument(
        "--name", default=""
    )  # will be saved as "{output_dir}/{name}.npy"

    args = parser.parse_args()

    if args.input_dir == None:
        print("[error] Please specify the input directory")
        exit(1)

    return args


def extract(files, args, model, mode=""):
    """
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    """
    assert len(files)

    print(f"Number of {mode} files: {len(files)}")

    segments, ans = model.prepare_data(files, args.task, int(args.max_len))
    if args.task == "custom":
        output_file = os.path.join(args.output_dir, f"{args.name}.npy")

    np.save(output_file, segments)
    print(f"Data shape: {segments.shape}, saved at {output_file}")


if __name__ == "__main__":
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = CP(dict=args.dict)
    files = glob(f"{args.input_dir}/*.mid")

    if args.task == "custom":
        extract(files, args, model)
