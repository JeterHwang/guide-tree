import torch
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.embedding import mbed
from src.upgma import UPGMA
from src.kmeans import BisectingKmeans
from src.utils import parse_aux, parseFile
from src.esm_github import pretrained
from src.prose.models.multitask import ProSEMT
from src.prose.models.lstm import SkipLSTM

def same_seed(seed):
    # Set seed for reproduciability
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

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
    with open(args.numpy_ckpt / 'test.npy', 'wb') as f:
        np.save(f, np.array(X))
        np.save(f, np.array(Y))
        np.save(f, np.array(Z))

def cluster_one_file(inputFile, outputFile, embedding, model, max_cluster_size, device, toks_per_batch):
        
    if embedding in ['esm', 'mBed']:
        Embedding = mbed(inputFile, embedding, model, device, toks_per_batch)
        sequences = Embedding.seqs
        centers, clusters = BisectingKmeans(sequences, device, max_cluster_size)
        #compare_Kmeans(args.compare, clusters)
    
    if embedding == 'esm':
        if len(centers) < 2:
            preCluster = UPGMA(clusters[0], 'AVG', 'L2_norm')
        else:
            preCluster = UPGMA(centers ,'AVG', 'L2_norm', 'clustal')
            for clusterID, cluster in enumerate(clusters):
                subtree = UPGMA(cluster, 'AVG', 'L2_norm', 'clustal')
                preCluster.appendTree(subtree, clusterID)    
    elif embedding == 'mBed':
        if len(centers) < 2:
            preCluster = UPGMA(clusters[0], 'AVG', 'K-tuple', 'clustal')
        else:
            preCluster = UPGMA(centers ,'AVG', 'Euclidean', 'clustal')
            for clusterID, cluster in enumerate(clusters):
                subtree = UPGMA(cluster, 'AVG', 'K-tuple', 'clustal')
                preCluster.appendTree(subtree, clusterID)
    elif embedding in ['prose_mt', 'prose_dlm']:
        preCluster = UPGMA(parseFile(inputFile), 'AVG', 'SSA', 'LCP', model)
    else:
        raise NotImplementedError
    
    preCluster.writeTree(outputFile)

def main(args):
    same_seed(args.seed)    
    device = torch.device(args.device)
    
    args.outputFolder.mkdir(parents=True, exist_ok=True)
    args.numpy_ckpt.mkdir(parents=True, exist_ok=True)
    
    if args.embedding == 'prose_mt':
        model = ProSEMT.load_pretrained(args.ckpt_path)
    elif args.embedding == 'prose_dlm':
        model = SkipLSTM.load_pretrained(args.ckpt_path)
    elif args.embedding == 'esm1_43M':
        model = pretrained.esm1_t6_43M_UR50S(args.ckpt_path) # model, alphabet
    elif args.embedding == 'esm1b_650M':
        model = pretrained.esm1b_t33_650M_UR50S(args.ckpt_path) # model, alphabet
    else:
        raise NotImplementedError

    for i, fastaFile in enumerate(list(args.inputFolder.glob('**/*.tfa'))):
        print(f"Now processing file ({i + 1}/{len(list(args.inputFolder.glob('**/*.tfa')))}) : {fastaFile.name}")
        cluster_one_file(fastaFile, args.outputFolder / f"{fastaFile.stem}_{args.embedding}.dnd", args.embedding, model, args.max_cluster_size, device, args.toks_per_batch)

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
        default="./output/bb3_release/prose_mt_100",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to pretrained protein embeddings.",
        default="./ckpt/prose/saved_models/prose_mt_3x1024.sav",
    )
    parser.add_argument(
        "--numpy_ckpt",
        type=Path,
        help="Path to save the numpy matrix/vector.",
        default="./ckpt/numpy",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Path to aux file",
        default='./output/BB50003/BB50003.aux'
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embedding", type=str, default='prose_mt', choices=['mBed', 'esm', 'prose_mt', 'prose_dlm'])
    parser.add_argument("--max_cluster_size", type=int, default=100)
    parser.add_argument("--toks_per_batch", type=int, default=4096)
    parser.add_argument("--UPGMA_type", type=str, choices=['LCP', 'clustal'], default='clustal')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
