from tqdm import tqdm
from Bio import SeqIO
import time
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='./astral-scopedom-seqres-gd-all-2.08-stable.fa')
    parser.add_argument('--query_size', type=int, default=10000)
    parser.add_argument('--target_size', type=int, default=10000)
    parser.add_argument('--query', type=Path, default='./query.fa')
    parser.add_argument('--target', type=Path, default='./target.fa')
    args = parser.parse_args()
    return args

def main(args):
    all_records = list(SeqIO.parse(args.input, 'fasta'))
    np.random.shuffle(all_records)
    with open(args.query, 'w') as f1, open(args.target, 'w') as f2:
        for i in range(args.query_size):
            id = all_records[i].id
            seq = all_records[i].seq
            f1.write(">" + str(id) + "\n")
            f1.write(str(seq) + "\n")
        for i in range(args.query_size, args.query_size + args.target_size):
            id = all_records[i].id
            seq = all_records[i].seq
            f2.write(">" + str(id) + "\n")
            f2.write(str(seq) + "\n")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)