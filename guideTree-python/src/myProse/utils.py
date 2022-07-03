from importlib.resources import path
import json
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO

def read_data(seq_path : Path, pair_path : Path):
    seqs = []
    for i, record in enumerate(SeqIO.parse(seq_path, 'fasta')):
        seqs.append(str(record.seq))
    pairs = []
    with open(pair_path, 'r') as f:
        for line in tqdm(f, desc=f"Reading {pair_path.name}"):
            line = json.loads(line)
            pairs.append(line)
    return seqs, pairs

