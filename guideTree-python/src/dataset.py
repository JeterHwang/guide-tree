import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset

class esmDataset(Dataset):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
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
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches

class proseDataset(Dataset):
    def __init__(self, seqs, save_path):
        self.save_path = save_path
        self.name = [seq['name'] for seq in seqs]
        self.seqNum = len(seqs)
        self.datasize = self.seqNum * (self.seqNum - 1) // 2
        self.xmapping = [0] * self.datasize
        self.ymapping = [0] * self.datasize

        k = 0
        for i in range(self.seqNum):
            for j in range(i):
                self.xmapping[k] = i
                self.ymapping[k] = j
                k += 1

    def __len__(self):
        return self.datasize
    
    def __getitem__(self, idx):
        seq1_ID = self.xmapping[idx]
        seq2_ID = self.ymapping[idx]
        return torch.load(self.save_path / f"{self.name[seq1_ID].replace('/', '-')}.pt"), torch.load(self.save_path / f"{self.name[seq2_ID].replace('/', '-')}.pt"), seq1_ID, seq2_ID

class LSTMDataset(Dataset):
    def __init__(self, seqs, alphabet):
        self.seqs = [seq['data'] for seq in seqs]
        self.name = [seq['name'] for seq in seqs]
        self.alphabet = alphabet
        self.encode_seqs = [torch.from_numpy(alphabet.encode(seq.encode('utf-8').upper())) for seq in self.seqs]

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        return self.name[idx], self.encode_seqs[idx]

    