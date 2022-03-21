import json
import pickle

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

import random
import numpy as np


def main(args):
    # Set seed for reproduciability
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

   
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="Directory to input protein sequence file.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
