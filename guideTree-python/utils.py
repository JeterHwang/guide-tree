from distutils.errors import DistutilsExecError
from tkinter.tix import Tree
from typing import Dict, List, final
import numpy as np
from pyrsistent import v

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

def encode(sequence : str) -> List[int]:
    encoded = []
    for ch in sequence:
        encoded.append(AMINO_ACID_CODE[ch.upper()])

def mapping(Kmer : List[int]) -> int:
    result, power = 0, 1 
    for token in Kmer:
        result = result + token * power
        power = power * AMINO_ACID
    return result

def frag_rel_pos(a1 : int, b1 : int, a2 : int, b2 : int, K : int):
    return (a1 - b1 == a2 - b2 and a2 < a1) or (a2 + K - 1 < a1 and b2 + K - 1 < b1)

# Return align score of 2 seqs
def KtupleDist(
    seq1 : List[int], 
    seq2 : List[int], 
    K : int, 
    signif : int,
    window : int,
    gapPenalty : int,
) -> float:
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
    
    validDiags = {}
    sortedDiag = sorted(diagonals, key=lambda x : len(x), reverse=True)
    for i in range(signif):
        if len(sortedDiag[i]) > 0:
            index = sortedDiag[i][0][0] - sortedDiag[i][0][1] + length2 - 1
            for diag in range(max(index - window, 0), min(index + window + 1, length1 + length2 - 1)):
                validDiags.add(diag)
    # Remove invalid diagonals
    for i, diag in enumerate(diagonals):
        if i not in validDiags:
            diag = []
    
    # (score, id, i, j)
    fragments = {}
    displ = [None for i in range(length1 + length2 - 1)]
    for i in range(length1 - K + 1):
        Ktuple1 = mapping(seq1[i : i + K])
        for j in KtupleLoc2[Ktuple1]:
            diagIndex = i - j + length2 - 1
            newID = len(fragments)
            if len(diagonals[diagIndex]) != 0: # valid diagonal
                # find predecessor 
                index = len(fragments) - 1
                while index >= 0 and frag_rel_pos(i, j, fragments[index][2], fragments[index][3], K):
                    index = index - 1
                if index < 0:  # no matched predecessor
                    displ[diagIndex] = (K, newID, i, j)
                    fragments.add((K, newID, i, j))
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
                    fragments.add((new_score, newID, i, j))
    if len(fragments) == 0:
        final_score = 0
    else:
        final_score = fragments[-1][0] / min(length1, length2) * 100
    return (100 - final_score) / 100



# seeeds [input] : raw seed sequences(Alphabic form)
# nodes [input] : raw input sequences(Alphabic form)
def seq2vec(
    seeds : List[Dict], 
    nodes : List[Dict], 
    K : int, 
    signif : int,
    window : int,
    gapPenalty : int,
) -> np.ndarray:
    # ENCODING
    encSeeds, encNodes = [], []
    for seed in seeds:
        encSeeds.append(encode(seed.data))
    for node in nodes:
        encNodes.append(encode(node.data))

    distMatrix = []
    for node in encNodes:
        vec = []
        for seed in encSeeds:
            vec.append(KtupleDist(node, seed, K, signif, window, gapPenalty))
        distMatrix.append(vec)

    return np.array(distMatrix)    
    
    
def parseFile(filePath : str) -> List[Dict]:
    