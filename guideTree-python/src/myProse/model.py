from __future__ import print_function,division

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence

import os
from src.prose.utils import get_project_root


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

        n = hidden_dim*num_layers + nin
        if bidirectional:
            n = 2*hidden_dim*num_layers + nin

        self.proj = nn.Linear(n, nout)

    @staticmethod
    def load_pretrained(path='prose_dlm'):
        if path is None or path == 'prose_dlm':
            root = get_project_root()
            path = os.path.join(root, 'saved_models', 'prose_dlm_3x1024.sav')

        model = SkipLSTM(21, 21, 1024, 3)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
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

    def forward(self, x1, x2, len1, len2):
        one_hot1 = self.to_one_hot(x1)
        one_hot2 = self.to_one_hot(x2)
        x1_packed = pack_padded_sequence(one_hot1, len1, batch_first=True, enforce_sorted=False)
        x2_packed = pack_padded_sequence(one_hot2, len2, batch_first=True, enforce_sorted=False)

        for f in self.layers:
            output, (hidden, cell) = f(x1_packed)
            x1_packed = output
        emb1 = hidden

        for f in self.layers:
            output, (hidden, cell) = f(x2_packed)
            x2_packed = output
        emb2 = hidden

        return torch.cat((emb1, emb2, torch.abs(emb1 - emb2)), 1)
