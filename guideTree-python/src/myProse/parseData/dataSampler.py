import json
import time
from tqdm import tqdm
from Bio import SeqIO
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
import subprocess
from subprocess import PIPE

def runcmd(command):
    bash_command = command.split()
    ret = subprocess.run(bash_command, stdout=PIPE, stderr=PIPE)
    if ret.returncode == 0:
        return ret.stdout
    else:
        print("Error !!")
        return ret.stderr

def get_similarity(idA, seqA, idB, seqB):
    with open('tmp.fa', 'w') as f:
        f.write(f'>{idA}\n')
        f.write(seqA + '\n')
        f.write(f'>{idB}\n')
        f.write(seqB + '\n')
    runcmd("mafft --localpair --distout --quiet tmp.fa")
    with open('tmp.fa.hat2', 'r') as f:
        for i in range(5):
            line = f.readline()
        line = f.readline().strip().replace('\n', '')
        distance = float(line)
    return round(max(2 - distance, 0), 3)
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', type=Path, default='./uniref50')
    parser.add_argument('--size', type=int, default=10000)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=Path, default='./uniref50')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.output_dir / f"{args.split}.json", 'w') as f:
        for i, fastaFile in enumerate(list(args.input.glob('**/*.fa'))):
            start_time = time.time()
            seqs = list(SeqIO.parse(fastaFile, 'fasta'))
            print(f'Load {fastaFile.name} in {time.time() - start_time}(s)')
            pairs = []
            while len(pairs) < args.size:
                sampled_pair = random.sample(range(len(seqs)), 2)
                pairs.append(sampled_pair) 
            data = []
            for i, (index_A, index_B) in enumerate(tqdm(pairs)):
                id_A, seq_A = str(seqs[index_A].id), str(seqs[index_A].seq) 
                id_B, seq_B = str(seqs[index_B].id), str(seqs[index_B].seq)
                similarity = get_similarity(id_A, seq_A, id_B, seq_B)
                f.write(json.dumps({'A' : seq_A, 'B' : seq_B, 'score' : similarity}))
                f.write('\n')
            
if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    main(args)