from importlib.resources import path
import json
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO

def read_data(seq_path : Path):
    pairs = []
    with open(seq_path, 'r') as f:
        for line in tqdm(f, desc=f"Reading {seq_path.name}"):
            line = json.loads(line)
            pairs.append(line)
    return pairs

