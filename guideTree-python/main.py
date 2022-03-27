from dataclasses import dataclass
from email.policy import default
import torch
import random
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from .utils import parseFile
from .embedding import mbed
from .kmeans import BisectingKmeans
from .upgma import UPGMA

def main(args):
    # Set seed for reproduciability
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    Embedding = mbed(args.inputFile)
    sequences = Embedding.seqs
    centers, clusters = BisectingKmeans(sequences)

    preCluster = UPGMA(centers ,'AVG')
    for cluster in clusters:
        subtree = UPGMA(cluster, 'AVG')
        preCluster.appendTree(subtree)
    
    preCluster.writeTree(args.outputFile)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--inputFile",
        type=str,
        help="Directory to input protein sequence file.",
        default="./data/",
    )
    parser.add_argument(
        "--outputFile",
        type=str,
        help="Directory to output guide tree.",
        default="./output/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

