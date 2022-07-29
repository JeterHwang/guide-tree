import json
from tqdm import tqdm
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

def get_distance(idA, seqA, idB, seqB):
    with open('tmp.fa', 'w') as f:
        f.write(f'>{idA}\n')
        f.write(seqA + '\n')
        f.write(f'>{idB}\n')
        f.write(seqB + '\n')
    runcmd("mafft --globalpair --distout --quiet tmp.fa")
    with open('tmp.fa.hat2', 'r') as f:
        for i in range(5):
            line = f.readline()
        line = f.readline().strip().replace('\n', '')
        distance = float(line)
    return round(0.5 * distance, 3)
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', type=Path, default='./train_raw.json')
    parser.add_argument('--output', type=Path, default='./train.json')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.output, 'w') as fout, open(args.input, 'r') as fin:
        for line in tqdm(fin.readlines(), desc=f"Adding Needleman Score:"):
            line = json.loads(line)
            seqA = line['A']
            seqB = line['B']
            score = line['score']
            dist = get_distance('A', seqA, 'B', seqB)
            fout.write(json.dumps({'A' : seqA, 'B' : seqB, 'score' : score, 'distance' : dist}))
            fout.write('\n') 
                          
if __name__ == '__main__':
    args = parse_args()
    main(args)