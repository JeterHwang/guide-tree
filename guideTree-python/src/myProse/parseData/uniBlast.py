import json
import time
import math
from tqdm import tqdm
from Bio import SeqIO
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
import subprocess
from subprocess import PIPE

mapping = [0,0,1,1,2,2,3,3,3,3]

def runcmd(command):
    bash_command = command.split()
    ret = subprocess.run(bash_command, stdout=PIPE, stderr=PIPE)
    if ret.returncode == 0:
        return ret.stdout
    else:
        print("Error !!")
        return ret.stderr

def parse_seqs(file_path, level):
    dictionary = {}
    with open(file_path, 'r') as fin:        
        sId = ''
        classification = ''
        sSeq = ''
        for line in fin:
            if line.startswith(">"):
                if sSeq != '':
                    if classification in dictionary:
                        dictionary[classification].append([sId, sSeq])
                    else:
                        dictionary[classification] = [[sId, sSeq]]
                sId = line.strip()[1:].split()[0]
                classification = '.'.join(line.strip()[1:].split()[1].split('.')[:level+1])
                sSeq = ''
            else:
                sSeq += line.strip()
    return dictionary

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
    return round(5 * max(2 - distance, 0), 3)
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input', type=Path, default='../astral/astral-scopedom-seqres-gd-sel-gs-sc-fa-2.08.fa')
    parser.add_argument('--output_dir', type=Path, default='./')
    parser.add_argument('--db_path', type=str, default='./swissprot/swissprot')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.input, 'r') as fin:
        start_time = time.time()
        seqsFromFile = list(SeqIO.parse(fin, 'fasta'))
        print(f'Load {fin.name} in {time.time() - start_time}(s)')
        
        # total_seqs = random.sample(seqsFromFile, args.size)
        total_seqs = seqsFromFile
        print(len(total_seqs))
        train_size = int(args.size * 0.9)
        
        mapping, data_size = {}, 0
        for split in ['train', 'eval']:
            if split == 'train':
                seqs = total_seqs[:train_size]
            else:
                seqs = total_seqs[train_size:]
            with open(args.output_dir / f"{split}.json", 'w') as fout, tqdm(total=len(seqs), desc=f'Parsing {split}.json') as t:
                for i, raw_seq in enumerate(seqs):
                    t.update(1)
                    id, seq = str(raw_seq.id), str(raw_seq.seq)
                    if len(seq) > 2000:
                        continue
                    with open('tmp.fa', 'w') as f:
                        f.write(f">{id}\n")
                        f.write(f"{seq}\n")
                    runcmd(f"blastp -query tmp.fa -db {args.db_path} -num_threads 8 -outfmt 10 -out tmp.csv")
                    
                    scores, known_entry = {}, []
                    with open("tmp.csv", 'r') as f, open("entry.in", 'w') as ff:
                        for line in f:
                            entry = line.strip().split(',')[1]
                            score = float(line.strip().split(',')[-1])
                            scores[entry] = score
                            data_size += 1
                            if entry not in mapping.keys():
                                ff.write(f"{entry}\n")
                            else:
                                known_entry.append(entry)
                    ## Sequences which are already queried   
                    for entry in known_entry:
                        if len(mapping[entry]) > 2000:
                            continue
                        fout.write(json.dumps({'A' : seq, 'B' : mapping[entry], 'score' : scores[entry]}))
                        fout.write("\n")
                    ## Call blastdbcmd to query unknown entries
                    runcmd(f"blastdbcmd -db {args.db_path} -entry_batch entry.in -out fasta.out")
                    hit_seqs = list(SeqIO.parse('fasta.out', 'fasta'))
                    for hit_seq in hit_seqs:
                        mapping[str(hit_seq.id)] = str(hit_seq.seq)
                        if len(str(hit_seq.seq)) > 2000:
                            continue
                        fout.write(json.dumps({'A' : seq, 'B' : str(hit_seq.seq), 'score' : scores[str(hit_seq.id)]}))
                        fout.write("\n")
                    t.set_postfix({
                        "Dict Size" : len(mapping.keys()),
                        "Dataset Size" : data_size
                    })
        
if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    main(args)