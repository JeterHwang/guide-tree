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
from src.utils import parse_aux

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

    Embedding = mbed(args.inputFile, args.embedding, args.esm_ckpt, device)
    sequences = Embedding.seqs
    
    centers, clusters = BisectingKmeans(sequences, device, args.max_cluster_size)
    mapping = parse_aux(args.compare)
    X, Y, Z = [], [], []
    for cluster in clusters:
        for seq in cluster:
            X.append(seq['embedding'])
            Y.append(seq['cluster'])
            Z.append(mapping[seq['name']])
            # print(seq['name'], seq['cluster'], mapping[seq['name']])
    
    # Save numpy checkpoint
    with open(args.numpy_ckpt, 'wb') as f:
        np.save(f, np.array(X))
        np.save(f, np.array(Y))
        np.save(f, np.array(Z))
    
    # testCluster = UPGMA(sequences, 'AVG', 'K-tuple')
    if len(centers) < 2:
        preCluster = UPGMA(clusters[0], 'AVG', 'K-tuple')
    else:
        preCluster = UPGMA(centers ,'AVG', 'Euclidean')
        for clusterID, cluster in enumerate(clusters):
            subtree = UPGMA(cluster, 'AVG', 'K-tuple')
            preCluster.appendTree(subtree, clusterID)
    
    preCluster.writeTree(args.outputFile)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--inputFile",
        type=str,
        help="Directory to input protein sequence file.",
        default="./data/bb3_release/RV30/BB30003.tfa",
    )
    parser.add_argument(
        "--outputFile",
        type=str,
        help="Path to output guide tree.",
        default="./output/BB30003/BB30003-python.dnd",
    )
    parser.add_argument(
        "--esm_ckpt",
        type=str,
        help="Path to pretrained protein embeddings.",
        default="./ckpt/esm/esm1_t6_43M_UR50S.pt",
    )
    parser.add_argument(
        "--numpy_ckpt",
        type=str,
        help="Path to save the numpy matrix/vector.",
        default="./ckpt/numpy/test.npy",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Path to aux file",
        default='./output/BB30003/BB30003.aux'
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embedding", type=str, default='mBed')
    parser.add_argument("--max_cluster_size", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
