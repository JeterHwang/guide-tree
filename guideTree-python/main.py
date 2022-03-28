import torch
import random
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import time

from src.embedding import mbed
from src.kmeans import BisectingKmeans
from src.upgma import UPGMA

def main(args):
    # Set seed for reproduciability
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    device = torch.device(args.device)

    Embedding = mbed(args.inputFile, args.embedding)
    sequences = Embedding.seqs
    centers, clusters = BisectingKmeans(sequences, device, 4)
    print()
    print()
    print()
    print()
    X, Y = [], []
    for cluster in clusters:
        for seq in cluster:
            X.append(seq['embedding'])
            Y.append(seq['cluster'])
            print(seq['name'], seq['cluster'])
    
    # Save numpy checkpoint
    with open(args.numpy_ckpt, 'wb') as f:
        np.save(f, np.array(X))
        np.save(f, np.array(Y))
    
    preCluster = UPGMA(centers ,'AVG', 'Euclidean')
    for cluster in clusters:
        subtree = UPGMA(cluster, 'AVG', 'K-tuple')
        preCluster.appendTree(subtree)
    
    preCluster.writeTree(args.outputFile)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--inputFile",
        type=str,
        help="Directory to input protein sequence file.",
        default="./data/bb3_release/RV12/BB12003.tfa",
    )
    parser.add_argument(
        "--outputFile",
        type=str,
        help="Directory to output guide tree.",
        default="./output/",
    )
    parser.add_argument(
        "--numpy_ckpt",
        type=str,
        help="Directory to save the model file.",
        default="./ckpt/numpy/test.npy",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embedding", type=str, default='pytorch')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
