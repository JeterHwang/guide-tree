import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence

import os

class SkipLSTM(nn.Module):
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0, bidirectional=True):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = nin
        for i in range(num_layers):
            f = nn.LSTM(
                dim, 
                hidden_dim, 
                1, 
                batch_first=True, 
                bidirectional=bidirectional
            )
            self.layers.append(f)
            if bidirectional:
                dim = 2*hidden_dim
            else:
                dim = hidden_dim

        n = hidden_dim*num_layers
        if bidirectional:
            n = 2*hidden_dim*num_layers

        self.linear = nn.Linear(n, nout)
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    @staticmethod
    def load_pretrained(path='prose_dlm'):
        model = SkipLSTM(21, 768, 1024, 3)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        return model

    def to_one_hot(self, x):
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def forward(self, x1, len1, x2, len2):
        one_hot1 = self.to_one_hot(x1)
        one_hot2 = self.to_one_hot(x2)
        x1_packed = pack_padded_sequence(one_hot1, len1, batch_first=True, enforce_sorted=False)
        x2_packed = pack_padded_sequence(one_hot2, len2, batch_first=True, enforce_sorted=False)

        hs1 = []
        for f in self.layers:
            output, (hidden, cell) = f(x1_packed)
            hs1 += [hidden[-1], hidden[-2]]
            x1_packed = output
        emb1 = torch.cat(hs1, dim=1)
        emb1 = self.linear(emb1)
        unsorted_indices1 = x1_packed.unsorted_indices
        emb1 = emb1.index_select(0, unsorted_indices1)

        hs2 = []
        for f in self.layers:
            output, (hidden, cell) = f(x2_packed)
            hs2 += [hidden[-1], hidden[-2]]
            x2_packed = output
        emb2 = torch.cat(hs2, dim=1)
        emb2 = self.linear(emb2)
        unsorted_indices2 = x2_packed.unsorted_indices
        emb2 = emb2.index_select(0, unsorted_indices2)

        return 1 + self.similarity(emb1, emb2)
