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
        esm_alphabet=None,
        alphabet=Uniprot21()
    ):
        self.split = split
        self.distance_metric = distance_metric
        self.seqs = read_data(seq_path / f"{self.split}.json")
        if esm_alphabet is not None:
            self.batch_converter = esm_alphabet.get_batch_converter()
        self.alphabet = alphabet
        self.seqA = []
        self.seqB = []
        self.distance = []
        for seq in self.seqs:
            if esm_alphabet is not None:
                self.seqA.append(seq['A'][:510].upper())
                self.seqB.append(seq['B'][:510].upper())
            else:
                self.seqA.append(torch.from_numpy(alphabet.encode(seq['A'][:510].encode('utf-8').upper())).long())
                self.seqB.append(torch.from_numpy(alphabet.encode(seq['B'][:510].encode('utf-8').upper())).long())
            
            if self.distance_metric == 'distance':
                self.distance.append(torch.tensor(min(max(float(seq['distance']), 0.0), 1.0)))
            elif self.distance_metric == 'score':
                self.distance.append(torch.tensor(float(seq['score'])))
            else:
                raise NotImplementedError    
        print('# loaded', len(self.seqs), 'sequence pairs')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        flip = random.choice([0,1])
        if flip:
            return self.seqA[idx], self.seqB[idx], self.distance[idx]
        else:
            return self.seqB[idx], self.seqA[idx], self.distance[idx]
    
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
        if hasattr(self, 'batch_converter'):
            seqA = [('A', sample[0]) for sample in samples] 
            seqB = [('B', sample[1]) for sample in samples] 
            seqs = seqA + seqB
            _, _, batch_tokens = self.batch_converter(seqs)  
        else:
            seqA = [sample[0] for sample in samples] 
            seqB = [sample[1] for sample in samples] 
            seqs = seqA + seqB
            padded = []
            for seq in seqs:
                padding = torch.tensor([-1] * max(0, 512 - len(seq)))
                padded.append(torch.cat([seq, padding], dim=0))
            batch_tokens = torch.stack(padded, dim=0).long()
        score = [sample[2] for sample in samples]   
        return batch_tokens, torch.tensor(score)

class LSTMDataset:
    def __init__(self, seqs, alphabet=Uniprot21()):
        self.seqs = seqs
        self.alphabet = alphabet
        if not isinstance(self.alphabet, Uniprot21):
            self.batch_converter = self.alphabet.get_batch_converter()
            self.tokens = [seq[:510] for seq in self.seqs]
        else:
            self.tokens = [torch.from_numpy(alphabet.encode(seq[:512].encode('utf-8').upper())).long() for seq in self.seqs]
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.tokens[idx]
    
    def collate_fn(self, samples):
        if hasattr(self, 'batch_converter'):
            seqs = [('A', sample) for sample in samples]
            _, _, batch_tokens = self.batch_converter(seqs)
        else:
            seqs = [sample for sample in samples]
            padded = []
            for seq in seqs:
                padding = torch.tensor([-1] * max(0, 512 - len(seq)))
                padded.append(torch.cat([seq, padding], dim=0))
            batch_tokens = torch.stack(padded, dim=0).long()
        return batch_tokens