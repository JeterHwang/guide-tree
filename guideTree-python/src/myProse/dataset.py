import torch
import random
from alphabets import Uniprot21
from utils import read_data
from torch.nn.utils.rnn import pad_sequence

class SCOPePairsDataset:
    def __init__(
        self, 
        seq_path,
        split,
        distance_metric='distance',
        alphabet=Uniprot21()
    ):
        self.split = split
        self.distance_metric = distance_metric
        self.seqs = read_data(seq_path / f"{self.split}.json")
        self.seqA = []
        self.seqB = []
        self.distance = []
        for seq in self.seqs:
            self.seqA.append(torch.from_numpy(alphabet.encode(seq['A'].encode('utf-8').upper())))
            self.seqB.append(torch.from_numpy(alphabet.encode(seq['B'].encode('utf-8').upper())))
            if self.distance_metric == 'distance':
                self.distance.append(torch.tensor(min(max(float(seq['distance']), 0.0), 1.0)))
            elif self.distance_metric == 'score':
                self.distance.append(torch.tensor(min(25 / float(seq['score']), 1.0)))
            else:
                raise NotImplementedError    
        self.alphabet = alphabet
        print('# loaded', len(self.seqs), 'sequence pairs')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        flip = random.choice([0,1])
        if flip:
            return self.seqA[idx].long(), self.seqB[idx].long(), self.distance[idx]
        else:
            return self.seqB[idx].long(), self.seqA[idx].long(), self.distance[idx]
    
    def batch_sampler(self, toks_per_batch=4096):
        sizes = [(len(s['A']) + len(s['B']), i) for i, s in enumerate(self.seqs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        
        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)
        
        _flush_current_buf()
        return batches
    
    def collate_fn(self, samples):
        seqA = [sample[0] for sample in samples] 
        seqB = [sample[1] for sample in samples] 
        score = [sample[2] for sample in samples]
        seqs = seqA + seqB
        lens = [len(x) for x in seqs]
        seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
        return seqs, lens, torch.tensor(score)

class LSTMDataset:
    def __init__(self, seqs, alphabet=Uniprot21()):
        self.seqs = seqs
        self.alphabet = alphabet
        self.tokens = [torch.from_numpy(alphabet.encode(seq.encode('utf-8').upper())) for seq in self.seqs]
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.tokens[idx].long()
    
    def batch_sampler(self, toks_per_batch=4096):
        sizes = [(len(s), i) for i, s in enumerate(self.seqs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        
        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)
        
        _flush_current_buf()
        return batches
    
    def collate_fn(self, samples):
        seqs = [sample for sample in samples] 
        lens = [len(x) for x in seqs]
        seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
        return seqs, lens