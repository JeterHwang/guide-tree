from audioop import reverse
from typing import Dict, List
from math import log2

from .utils import seq2vec

Ktuple_param = {
    "K" : 1, 
    "signif" : 3,
    "window" : 5,
    "gapPenalty" : 5
}

class mbed(object):
    
    def __init__(self, seqs : List[Dict]) -> None:
        self.sort_by_length(seqs)
        self.nseq = len(seqs)
        self.istep = int(self.nseq / self.seedNumber)
        self.mBed = seq2vec(self.seqs, self.seed, **Ktuple_param)
        
    def sort_by_length(self, seqs):
        self.seqs =  sorted(seqs, key=lambda seq: seq.length, reverse=True)
    
    @property
    def numSeed(self):
        return int(log2(self.nseq)**2)
    
    @property
    def seed(self):
        seedList = []
        for i in range(0, self.numSeed, self.istep):
            seedList.append(self.seqs[i])
        seedList = sorted(seedList, key=lambda seed: seed.id, reverse=True)
        return seedList
    