import time
from typing import Dict, List
import threading
from math import sqrt
import numpy as np
import subprocess
from subprocess import PIPE

import torch
from torch.utils.data import DataLoader

from src.threading import Worker

from src.dataset import esmDataset
from src.embed_sequences import prose_embedding, SSA_score
BIG_DIST = 1e29
AMINO_ACID = 25
AMINO_ACID_CODE = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
    "E" : 4,
    "F" : 5,
    "G" : 6,
    "H" : 7,
    "I" : 8,
    "K" : 9,
    "L" : 10,
    "M" : 11,
    "N" : 12,
    "P" : 13,
    "Q" : 14,
    "R" : 15,
    "S" : 16,
    "T" : 17,
    "U" : 18,
    "V" : 19,
    "W" : 20,
    "X" : 21,
    "Y" : 22,
    "Z" : 23,
    "-" : 24,
}

__all__ = [
    'encode',
    'mapping',
    'frag_rel_pos',
    'KtupleDist',
    'seq2vec',
    'parseFile',
    'Euclidean',
    'distMatrix',
]

def encode(sequence : str) -> List[int]:
    encoded = []
    for ch in sequence:
        encoded.append(AMINO_ACID_CODE[ch.upper()])
    return encoded

def mapping(Kmer : List[int]) -> int:
    result, power = 0, 1 
    for token in Kmer:
        result = result + token * power
        power = power * AMINO_ACID
    return result

def put_frag(fragments : List[tuple], newElement : tuple) -> None:
    for i in range(len(fragments) - 1, -1, -1):
        if fragments[i][0] < newElement[0]:
            return fragments[:i+1] + [newElement] + fragments[i+1:]
    return [newElement] + fragments

def frag_rel_pos(a1 : int, b1 : int, a2 : int, b2 : int, K : int):
    return (a1 - b1 == a2 - b2 and a2 < a1) or (a2 + K - 1 < a1 and b2 + K - 1 < b1)

# Return align score of 2 seqs
def KtupleDist(
    seq1str : str, 
    seq2str : str, 
    K = 1, 
    signif = 5,
    window = 5,
    gapPenalty = 3,
) -> float:
    seq1, seq2 = encode(seq1str), encode(seq2str)
    length1, length2 = len(seq1), len(seq2)
    # The location of each kind of K-tuple match 
    KtupleLoc2 = [[] for i in range(AMINO_ACID**K)]
    
    # The number of matched K-tuple in each diagonal
    diagonals = [[] for i in range(length1 + length2 - 1)]
    
    for i in range(length2 - K + 1):
        Ktuple2 = mapping(seq2[i : i + K])
        KtupleLoc2[Ktuple2].append(i)
    
    for i in range(length1 - K + 1):
        Ktuple1 = mapping(seq1[i : i + K])
        for j in KtupleLoc2[Ktuple1]:
            diagonals[i - j + length2 - 1].append((i, j))
    validDiags = set()
    sortedDiag = sorted(diagonals, key=lambda x : len(x), reverse=True)
    for i in range(signif):
        if len(sortedDiag[i]) > 0:
            index = sortedDiag[i][0][0] - sortedDiag[i][0][1] + length2 - 1
            for diag in range(max(index - window, 0), min(index + window + 1, length1 + length2 - 1)):
                validDiags.add(diag)
    # Remove invalid diagonals
    for i in range(len(diagonals)):
        if i not in validDiags:
            diagonals[i] = []
    
    # (score, id, i, j)
    fragments = []
    displ = [None for i in range(length1 + length2 - 1)]
    max_aln_length = max(max(length1, length2) * 2, AMINO_ACID**K + 1)
    for i in range(length1 - K + 1):
        # Early Stopping criterion set by Clustal Omega
        if len(fragments) >= max_aln_length * 2:
            #print('Partial alignment !!')
            break
        
        Ktuple1 = mapping(seq1[i : i + K])
        for j in KtupleLoc2[Ktuple1]:
            # Early Stopping criterion set by Clustal Omega
            if len(fragments) >= max_aln_length * 2:
                break
            
            diagIndex = i - j + length2 - 1
            newID = len(fragments)
            if len(diagonals[diagIndex]) != 0: # valid diagonal
                # find predecessor 
                index = len(fragments) - 1
                while index >= 0 and not frag_rel_pos(i, j, fragments[index][2], fragments[index][3], K):
                    index = index - 1
                if index < 0:  # no matched predecessor
                    displ[diagIndex] = (K, newID, i, j)
                    fragments = put_frag(fragments, (K, newID, i, j))
                else:
                    predecessor = fragments[index]
                    if i - j == predecessor[2] - predecessor[3]: # on the same diagonal
                        new_score = predecessor[0] + K if i > predecessor[2] + K - 1 else predecessor[0] + i - predecessor[2]
                    else:   # on different diagonal
                        subt2 = predecessor[0] - gapPenalty + K
                        if displ[diagIndex] is None:
                            new_score = max(K, subt2)
                        elif i > displ[diagIndex][2] + K - 1:
                            new_score = max(displ[diagIndex][0] + K, subt2)
                        else:
                            new_score = max(displ[diagIndex][0] + i - displ[diagIndex][2])
                    displ[diagIndex] = (new_score, newID, i, j)
                    fragments = put_frag(fragments, (new_score, newID, i, j))
            # Sort the set whenever a new element is added
            # fragments = sorted(fragments, key=lambda x : x[0])

    if len(fragments) == 0:
        final_score = 0
    else:
        final_score = fragments[-1][0] / min(length1, length2) * 100
    
    return (100 - final_score) / 100

def Ktupledist_multiThread(seqs, seeds, K, signif, window, gapPenalty, threadName):
    for i, seq in enumerate(seqs):
        vec = []
        for seed in seeds:
            vec.append(KtupleDist(seq['data'], seed['data'], K, signif, window, gapPenalty))
        if seqs[i]['embedding'] == None:
            seqs[i]['embedding'] = np.array(vec)

# seeeds [input] : raw seed sequences(Alphabic form)
# nodes [input] : raw input sequences(Alphabic form)
def seq2vec(
    seqs : List[Dict], 
    seeds : List[Dict], 
    convertType : str,
    K : int, 
    signif : int,
    window : int,
    gapPenalty : int,
    model = None,
    device = None,
    toks_per_batch = 4096,
    multi_threading=False,
    num_threads = 6,
    save_path = None,
) -> np.ndarray:
    start_time = time.time()
    #print('----- Start converting sequences to vector -----')
    if convertType == 'mBed':
        if multi_threading:
            chunkLength = len(seqs) // num_threads if len(seqs) % num_threads == 0 else len(seqs) // num_threads + 1
            thread_list = []
            for i in range(0, len(seqs), chunkLength):
                thread_list.append(threading.Thread(
                    target=Ktupledist_multiThread, 
                    name=f'Thread-{i // chunkLength}', 
                    args=(seqs[i : i + chunkLength], seeds, K, signif, window, gapPenalty, f'Thread-{i // chunkLength}')
                ))
                thread_list[-1].start()
            for i in range(len(thread_list)):
                thread_list[i].join()
        else:
            for i, seq in enumerate(seqs):
                vec = []
                for seed in seeds:
                    vec.append(KtupleDist(seq['data'], seed['data'], K, signif, window, gapPenalty))
                if seqs[i]['embedding'] == None:
                    seqs[i]['embedding'] = np.array(vec)
    elif convertType == 'esm':
        repr = esm_embedding([seq['name'] for seq in seqs], [seq['data'] for seq in seqs], model, device, toks_per_batch)
        assert len(repr) == len(seqs)
        for i, emb in enumerate(repr):
            seqs[i]['embedding'] = emb.cpu().detach().numpy()
    elif convertType in ['prose_mt', 'prose_dlm']:
        repr = prose_embedding(seqs, model, 'avg', toks_per_batch, save_path)
        assert len(repr) == len(seqs)
        for i, emb in enumerate(repr):
            seqs[i]['embedding'] = emb
    else:
        raise NotImplementedError
    #print(f'----- Finish in {time.time() - start_time} seconds -----')

    
def parseFile(filePath : str) -> List[Dict]:
    returnData = []
    with open(filePath, 'r') as f:
        id = 0
        name = f.readline().replace('\n', '')
        while name and name[0] == '>':
            data = ''
            while True:
                line = f.readline()
                if not line or line[0] == '>':
                    break
                else:
                    data =  data + line.replace('\n', '')
            returnData.append({
                'name' : name[1:].replace('\n', ''),
                'data' : data,
                'id' : id,
                'embedding' : None,
                'cluster' : 0
            })
            id = id + 1
            name = line
    if len(returnData) == 0:
        raise NotImplementedError
    else:
        return returnData 

def Euclidean(P1 : np.ndarray, P2 : np.ndarray) -> float:
    return np.dot(P1 - P2, P1 - P2)

def L2_norm(P1 : np.ndarray, P2 : np.ndarray) -> float:
    P1_norm = P1 / sqrt(np.dot(P1, P1))
    P2_norm = P2 / sqrt(np.dot(P2, P2))
    return np.dot(P1_norm - P2_norm, P1_norm - P2_norm)

def Cosine(P1 : np.ndarray, P2 : np.ndarray) -> float:
    return 1 - np.dot(P1, P2) / (sqrt(np.dot(P1, P1))) / (sqrt(np.dot(P2, P2)))

def distMatrix(Nodes : List[Dict], dist_type : str, model=None, save_path=None) -> np.ndarray:
    nodeNum = len(Nodes)
    if dist_type == 'SSA':
        assert model is not None
        Matrix = SSA_score(Nodes, model, save_path)
    else:
        Matrix = np.full((nodeNum, nodeNum), BIG_DIST)
        for i in range(nodeNum):
            for j in range(i):
                if dist_type == 'Euclidean':
                    Matrix[i][j] = Matrix[j][i] = Euclidean(Nodes[i]['embedding'], Nodes[j]['embedding'])
                elif dist_type == 'Cosine':
                    Matrix[i][j] = Matrix[j][i] = Cosine(Nodes[i]['embedding'], Nodes[j]['embedding'])
                elif dist_type == 'L2_norm':
                    Matrix[i][j] = Matrix[j][i] = L2_norm(Nodes[i]['embedding'], Nodes[j]['embedding'])
                elif dist_type == 'K-tuple':
                    Matrix[i][j] = Matrix[j][i] = KtupleDist(Nodes[i]['data'], Nodes[j]['data'])
                else:
                    raise NotImplementedError
    return Matrix

def esm_embedding(labels, sequences, model_alphabet, device, toks_per_batch, truncate=False):
    # Load ESM-1b model
    model, alphabet = model_alphabet[0], model_alphabet[1]
    model.eval()  # disables dropout for deterministic results
    model = model.to(device)

    dataset = esmDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(), 
        batch_sampler=batches,
        pin_memory=True
    )
    
    #### decide repr_layers by model configuration ####
    if ckpt_path.stem == 'esm1_t6_43M_UR50S':         #
        repr_layers = [6]                             #
    elif ckpt_path.stem == 'esm1b_t33_650M_UR50S':    #
        repr_layers = [33]  ## original [0, 32, 33]   #
    else:                                             #
        raise NotImplementedError                     #
    ###################################################
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    
    # Extract per-residue representations (on CPU)
    sequence_representations = []
    with torch.no_grad():
        for idx, (names, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            # if truncate:
            #     toks = toks[:, :1022]

            toks = toks.to(device, non_blocking=True)
            output = model(toks, repr_layers=repr_layers, return_contacts=False)
            logits = output['logits'].to(device='cpu')
            representations = output['representations'][repr_layers[0]]

            # Mean mode
            for i, seq in enumerate(strs):                
                sequence_representations.append(representations[i, 1 : len(seq) + 1].mean(0).clone())
    return sequence_representations

def parse_aux(aux_file) -> Dict:
    mapping = {}
    with open(aux_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()
            mapping[line[8]] = int(line[1].replace(':', ''))
    return mapping

def runcmd(command):
    bash_command = command.split()
    ret = subprocess.run(bash_command, stdout=PIPE, stderr=PIPE)
    if ret.returncode == 0:
        return ret.stdout
    else:
        print("Error !!")
        return ret.stderr