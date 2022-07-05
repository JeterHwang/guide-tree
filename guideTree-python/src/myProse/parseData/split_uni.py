import json
from tqdm import tqdm
from Bio import SeqIO
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
import subprocess
from subprocess import PIPE

def write_file(sId, sSeq, output_files, len_list):
    length = len(sSeq)
    for file, maxx in zip(output_files, len_list):
        if length <= maxx:
            file.write(f">{sId}\n")
            file.write(f"{sSeq}\n")
            break
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', type=Path, default='./uniref50.fasta')
    parser.add_argument('--output_dir', type=Path, default='./uniref50/')
    parser.add_argument('--lengths', type=list, default=[200,400,600,800,1000])
    args = parser.parse_args()
    return args

def main(args):
    with open(args.input, 'r') as fin:
        output_files = []
        for length in args.lengths:
            f = open(args.output_dir / f"uniref50_{length}.fa", 'w')
            output_files.append(f)
        with tqdm(total=52523202, desc='Processing Uniref50') as t:
            sId = ''
            sSeq = ''
            for line in fin:
                if line.startswith(">"):
                    if sSeq != '':
                        t.update(1)
                        write_file(sId, sSeq, output_files, args.lengths)    
                    sId = line.strip()[1:].split()[0]
                    sSeq = ''
                else:
                    sSeq += line.strip()
            write_file(sId, sSeq, output_files, args.lengths)    
        for file in output_files:
            file.close()
    
if __name__ == '__main__':
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)