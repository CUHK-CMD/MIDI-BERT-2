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
        choices=["custom", "skyline"],
    )

    ### path ###
    parser.add_argument("--dict", type=str, default="../dict/CP_program.pkl")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--input_txt", type=str, default=None)

    ### parameter ###
    parser.add_argument("--max_len", type=int, default=512)

    ### output ###
    parser.add_argument("--output_dir", default="../data/CP")
    parser.add_argument(
        "--name", default=""
    )  # will be saved as "{output_dir}/{name}.npy"

    args = parser.parse_args()

    if args.input_dir == None and args.input_txt == None:
        print("[error] Please specify input_dir or input_txt")
        exit(1)
    if args.input_dir != None and args.input_txt != None:
        print("[error] Either specify input_dir or input_txt")
        exit(1)

    return args


if __name__ == "__main__":
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.input_dir is not None:
        files = [
            os.path.join(path, name)
            for path, subdir, files in os.walk(args.input_dir)
            for name in files
            if name.endswith("mid")
        ]
    else:
        with open(args.input_txt) as f:
            files = [line.strip() for line in f.readlines()]
    assert len(files)
    print(f"Number of files: {len(files)}")

    model = CP(args.dict, files, args.task, int(args.max_len))
    segments, ans = model.prepare_data()
    if args.task == "custom" or args.task == "skyline":
        output_file = os.path.join(args.output_dir, f"{args.name}.npy")
        np.save(output_file, segments)
        print(f"Data shape: {segments.shape}, saved at {output_file}")

    if args.task == "skyline":
        ans_file = os.path.join(args.output_dir, f"{args.name}_ans.npy")
        np.save(ans_file, ans)
        print(f"Answer shape: {ans.shape}, saved at {ans_file}")
