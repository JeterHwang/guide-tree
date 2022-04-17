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

def same_seed(seed):
    # Set seed for reproduciability
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compare_Kmeans(compareFile, clusters):
    mapping = parse_aux(compareFile)
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

def cluster_one_file(inputFile, outputFile, embedding, esm_ckpt, max_cluster_size, device):
    Embedding = mbed(inputFile, embedding, esm_ckpt, device)
    sequences = Embedding.seqs
    
    centers, clusters = BisectingKmeans(sequences, device, max_cluster_size)
    #compare_Kmeans(args.compare, clusters)
    
    if len(centers) < 2:
        preCluster = UPGMA(clusters[0], 'AVG', 'K-tuple')
    else:
        preCluster = UPGMA(centers ,'AVG', 'Euclidean')
        for clusterID, cluster in enumerate(clusters):
            subtree = UPGMA(cluster, 'AVG', 'K-tuple')
            preCluster.appendTree(subtree, clusterID)
    
    preCluster.writeTree(outputFile)

def main(args):
    same_seed(args.seed)    
    device = torch.device(args.device)
    
    args.outputFolder.mkdir(parents=True, exist_ok=True)
    for fastaFile in list(args.inputFolder.glob('**/*.tfa')):
        cluster_one_file(fastaFile, args.outputFolder / f"{fastaFile.stem}.dnd", args.embedding, args.esm_ckpt, args.max_cluster_size, device)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--inputFolder",
        type=Path,
        help="Directory to input protein sequence file.",
        default="./data/bb3_release",
    )
    parser.add_argument(
        "--outputFolder",
        type=Path,
        help="Path to output guide tree.",
        default="./output/bb3_release",
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
        default='./output/BB50003/BB50003.aux'
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
