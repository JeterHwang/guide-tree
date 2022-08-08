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
        print(command)
        print(ret.stderr)
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

def get_distance(idA, seqA, idB, seqB, tmp_file):
    with open(tmp_file, 'w') as f:
        f.write(f'>{idA}\n')
        f.write(seqA + '\n')
        f.write(f'>{idB}\n')
        f.write(seqB + '\n')
    runcmd(f"mafft --anysymbol --globalpair --distout --quiet {str(tmp_file.absolute().resolve())}")
    with open(tmp_file.parent / f"{tmp_file.name}.hat2", 'r') as f:
        for i in range(5):
            line = f.readline()
        line = f.readline().strip().replace('\n', '')
        distance = float(line)
    return round(0.5 * distance, 3)
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input', type=Path, default='../astral/astral-scopedom-seqres-gd-sel-gs-sc-fa-2.08.fa')
    parser.add_argument('--output_dir', type=Path, default='./')
    parser.add_argument('--db_path', type=str, default='../uniprotKB/uniprotKB')
    parser.add_argument('--tmp_folder', type=Path, default='./tmp')
    args = parser.parse_args()
    return args

def main(args):
    args.tmp_folder.mkdir(parents=True, exist_ok=True)
    
    with open(args.input, 'r') as fin:
        start_time = time.time()
        seqsFromFile = list(SeqIO.parse(fin, 'fasta'))
        print(f'Load {fin.name} in {time.time() - start_time}(s)')
        
        # total_seqs = random.sample(seqsFromFile, args.size)
        total_seqs = seqsFromFile
        train_size = int(len(total_seqs) * 0.99)
        print(f"Train size : Eval size = {train_size} : {len(total_seqs) - train_size}")

        mapping, data_size = {}, 0
        for split in ['train', 'eval']:
            if split == 'train':
                seqs = total_seqs[:train_size]
            else:
                seqs = total_seqs[train_size:]
            with open(args.output_dir / f"{split}_raw.json", 'w') as fout, tqdm(total=len(seqs), desc=f'Parsing {split}_raw.json') as t:
                for i, raw_seq in enumerate(seqs):
                    t.update(1)
                    id, seq = str(raw_seq.id), str(raw_seq.seq)
                    if len(seq) > 2000:
                        continue
                    
                    tmp_fasta = args.tmp_folder / f"{id}.fa"
                    tmp_csv = args.tmp_folder / f"{id}.csv"
                    tmp_entry = args.tmp_folder / f"{id}.in"
                    tmp_dbout = args.tmp_folder / f"{id}.out"
                    
                    with open(tmp_fasta, 'w') as f:
                        f.write(f">{id}\n")
                        f.write(f"{seq}\n")
                    runcmd(f"blastp -query {str(tmp_fasta.absolute().resolve())} -db {args.db_path} -num_threads 8 -num_alignments 125 -outfmt 10 -out {str(tmp_csv.absolute().resolve())}")
                    
                    scores, known_entry, visited = {}, [], []
                    with open(tmp_csv, 'r') as f, open(tmp_entry, 'w') as ff:
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
                        if mapping[entry] in visited:
                            continue
                        visited.append(mapping[entry])
                        distance = get_distance('A', seq, 'B', mapping[entry], args.tmp_folder / f"{id}_{entry}.fa")
                        fout.write(json.dumps({'A' : seq, 'B' : mapping[entry], 'score' : scores[entry], 'distance' : distance}))
                        fout.write("\n")
                    ## Call blastdbcmd to query unknown entries
                    runcmd(f"blastdbcmd -db {args.db_path} -entry_batch {str(tmp_entry.absolute().resolve())} -out {str(tmp_dbout.absolute().resolve())}")
                    hit_seqs = list(SeqIO.parse(tmp_dbout, 'fasta'))
                    for hit_seq in hit_seqs:
                        entry, sequence = str(hit_seq.id), str(hit_seq.seq)
                        mapping[entry] = sequence
                        if len(sequence) > 2000:
                            continue
                        if sequence in visited:
                            continue
                        visited.append(sequence)
                        distance = get_distance('A', seq, 'B', sequence, args.tmp_folder / f"{id}_{entry}.fa")
                        fout.write(json.dumps({'A' : seq, 'B' : sequence, 'score' : scores[entry], 'distance' : distance}))
                        fout.write("\n")
                    t.set_postfix({
                        "Dict Size" : len(mapping.keys()),
                        "Dataset Size" : data_size
                    })
        
if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    main(args)
