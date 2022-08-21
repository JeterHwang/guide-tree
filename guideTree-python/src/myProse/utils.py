from importlib.resources import path
import json
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO

def read_data(seq_path : Path):
    pairs = []
    with open(seq_path, 'r') as f:
        for line in tqdm(f.readlines(), desc=f"Reading {seq_path.name}"):
            line = json.loads(line)
            seqA, seqB = line['A'], line['B']
            if len(seqA) <= 512 and len(seqB) <= 512 and abs(len(seqA) - len(seqB)) < 100:
                pairs.append(line)
    return pairs

