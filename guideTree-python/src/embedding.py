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
    
    def __init__(self, file, convertType='mBed', model=None, device=None, toks_per_batch=4096, save_path=None) -> None:
        self.seqs = parseFile(file)
        self.nseq = len(self.seqs)
        self.istep = int(self.nseq / self.numSeed)
        seq2vec(
            seqs = self.seqs, 
            seeds = self.seed, 
            convertType = convertType, 
            device = device,
            model = model,
            toks_per_batch=toks_per_batch,
            save_path = save_path,
            **Ktuple_param
        )

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
    