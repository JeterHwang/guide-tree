from audioop import reverse
from typing import Dict, List
from math import log2

from .utils import seq2vec, parseFile

Ktuple_param = {
    "K" : 1, 
    "signif" : 5,
    "window" : 5,
    "gapPenalty" : 3
}

__all__ = [
    'mbed',
]

class mbed(object):
    
    def __init__(self, file, convertType='mBed') -> None:
        self.seqs = parseFile(file)
        self.nseq = len(self.seqs)
        self.istep = int(self.nseq / self.numSeed)
        seq2vec(self.seqs, self.seed, convertType, **Ktuple_param)
        for seq in self.seqs:
            print(seq['name'], seq['embedding'])

    @property
    def sorted_seqs(self):
        return sorted(self.seqs, key=lambda seq: len(seq['data']), reverse=True)

    @property
    def numSeed(self):
        log2seed = int(log2(self.nseq)**2)
        return log2seed if log2seed < self.nseq else self.nseq - 1
    
    @property
    def seed(self):
        seedList = []
        for i in range(0, self.numSeed, self.istep):
            seedList.append(self.sorted_seqs[i])
        seedList = sorted(seedList, key=lambda seed: seed['id'])
        return seedList
    