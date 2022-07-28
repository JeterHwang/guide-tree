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
    runcmd("mafft --globalpair --distout --quiet tmp.fa")
    with open('tmp.fa.hat2', 'r') as f:
        for i in range(5):
            line = f.readline()
        line = f.readline().strip().replace('\n', '')
        distance = float(line)
    return round(0.5 * distance, 3)
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', type=Path, default='./uniprotKB_slicing')
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--level', type=int, default=10)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=Path, default='./uniprotKB_slicing')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.output_dir / f"{args.split}.json", 'w') as f:
        for i, fastaFile in enumerate(list(args.input.glob('**/*.fa'))):
            if 'xxxxx' in fastaFile.name:
                visited_ids = [[] for _ in range(args.level)]
                with tqdm(total=args.size * args.level, desc=f'Sample File {fastaFile.name}') as t:
                    for j in range(4):    
                        start_time = time.time()
                        seqs = parse_seqs(fastaFile, j)
                        print(f'Load {fastaFile.name} in {time.time() - start_time}(s)')
                    
                        total_pairs, discarded_pairs = 0, 0
                        classes = list(seqs.keys())
                        while total_pairs < args.size * args.level and (j == 3 or discarded_pairs < 2 * args.size * args.level):
                            cls = random.choice(classes)
                            if len(seqs[cls]) < 2:
                                continue
                            entry_A, entry_B = random.sample(seqs[cls], 2)
                            id_A, seq_A = str(entry_A[0]), str(entry_A[1]) 
                            id_B, seq_B = str(entry_B[0]), str(entry_B[1])
                            if abs(len(seq_A)-len(seq_B)) > 50:
                                continue
                            similarity = get_similarity(id_A, seq_A, id_B, seq_B)
                            lvl = math.floor(similarity)
                            
                            if lvl < args.level and len(visited_ids[lvl]) < args.size and [id_A, id_B] not in visited_ids[lvl]:
                                f.write(json.dumps({'A' : seq_A, 'B' : seq_B, 'score' : similarity}))
                                f.write('\n')
                                visited_ids[lvl].append([id_A, id_B])
                                total_pairs += 1    
                                t.update(1)
                                if len(visited_ids[lvl]) == args.size:
                                    print(f"Level {lvl} is full !!")
                            else:
                                discarded_pairs += 1
                            t.set_postfix({
                                "discarded" : discarded_pairs,
                                'hierachy' : j,
                            })
            else:
                start_time = time.time()
                seqs = list(SeqIO.parse(fastaFile, 'fasta'))
                print(f'Load {fastaFile.name} in {time.time() - start_time}(s)')
                total_pairs = 0
                with tqdm(total=args.size, desc=f'Sample File {fastaFile.name}') as t:    
                    while total_pairs < args.size:
                        entry_A, entry_B = random.sample(seqs, 2)
                        id_A, seq_A = str(entry_A.id), str(entry_A.seq) 
                        id_B, seq_B = str(entry_B.id), str(entry_B.seq) 
                        similarity = get_similarity(id_A, seq_A, id_B, seq_B)
                        f.write(json.dumps({'A' : seq_A, 'B' : seq_B, 'score' : similarity}))
                        f.write('\n')
                        total_pairs += 1    
                        t.update(1)
                        
if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    main(args)