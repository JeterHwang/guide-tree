from typing import Dict, List
from queue import PriorityQueue
import numpy as np
import torch
import esm.esm as esm

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
            print('Partial alignment !!')
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
) -> np.ndarray:
    if convertType == 'mBed':
        for i, seq in enumerate(seqs):
            vec = []
            for seed in seeds:
                vec.append(KtupleDist(seq['data'], seed['data'], K, signif, window, gapPenalty))
            if seqs[i]['embedding'] == None:
                seqs[i]['embedding'] = np.array(vec)
    elif convertType == 'pytorch':
        repr = esm_embedding([(seq['name'], seq['data']) for seq in seqs])
        assert len(repr) == len(seqs)
        for i, emb in enumerate(repr):
            seqs[i]['embedding'] = emb.cpu().detach().numpy()
    else:
        raise NotImplementedError
    
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
                'name' : name[1:],
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

def Euclidean(P1 : Dict, P2 : Dict) -> float:
    return np.dot(P1.embedding - P2.embedding, P1.embedding - P2.embedding)

def distMatrix(Nodes : List[Dict], dist_type : str) -> np.ndarray:
    nodeNum = len(Nodes)
    Matrix = np.zeros((nodeNum, nodeNum))
    for i in range(nodeNum):
        for j in range(i + 1):
            if dist_type == 'Euclidean':
                Matrix[i][j] = Euclidean(Nodes[i], Nodes[j])
            elif dist_type == 'K-tuple':
                Matrix[i][j] = KtupleDist(Nodes[i], Nodes[j])
            else:
                raise NotImplementedError
    return Matrix

def esm_embedding(data):
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1_t6_43M_UR50S('./ckpt/esm/esm1_t6_43M_UR50S.pt')
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    
    return sequence_representations