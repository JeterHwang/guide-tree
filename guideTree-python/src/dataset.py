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

class SSADataset(Dataset):
    def __init__(self, seq1, seq2, alphabet):
        assert len(seq1) == len(seq2)
        self.seq1 = [alphabet.encode(x.encode('utf-8').upper()) for x in seq1]
        self.seq2 = [alphabet.encode(x.encode('utf-8').upper()) for x in seq2]
        # for s1, s2 in zip(self.seq1, self.seq2):
        #     print(len(s1), len(s2))
    def __len__(self):
        return len(self.seq1)
    
    def __getitem__(self, idx):
        return self.seq1[idx], self.seq2[idx]
    
    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [len(s1) + len(s2) for s1, s2 in zip(self.seq1, self.seq2)]
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

        for i, sz in enumerate(sizes):
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches
    
    def collate_fn(self, batch_seqs):
        seq1 = [torch.LongTensor(seq_tuple[0]) for seq_tuple in batch_seqs]
        seq2 = [torch.LongTensor(seq_tuple[1]) for seq_tuple in batch_seqs]
        seqs = seq1 + seq2
        #print(seqs)
        seq_length = np.array([len(seq) for seq in seqs])
        order = np.argsort(seq_length)[::-1]
        max_len = max(seq_length)
        
        new_seq = seqs[0].new(len(seqs), max_len).zero_()
        for i in range(len(seqs)):
            j = order[i]
            X = seqs[j]
            new_seq[i, : len(X)] = X

        seq_length = seq_length[order]
        #print(new_seq)
        #print(order)
        packed_seqs = pack_padded_sequence(new_seq, seq_length, batch_first=True)
        return packed_seqs, order
        