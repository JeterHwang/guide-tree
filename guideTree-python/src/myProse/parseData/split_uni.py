from tqdm import tqdm
from Bio import SeqIO
import time
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='./astral-scopedom-seqres-gd-all-2.08-stable.fa')
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--output', type=str, default='./query.fa')
    args = parser.parse_args()
    return args

def main(args):
    start_time = time.time()
    all_records = list(SeqIO.parse(args.input, 'fasta'))
    print(len(all_records))
    np.random.shuffle(all_records)
    print(all_records[0])
    print(f"Elapsed Time : {time.time() - start_time}s")

if __name__ == '__main__':
    args = parse_args()
    main(args)