import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
import os

class L1(nn.Module):
    def forward(self, x, y, chunk_size=40000):
        if x.size(1) > 100:
            x_chunk_size = chunk_size // y.size(0) + 1
            L1_dis = torch.zeros(x.size(0), y.size(0), device=x.device)
            for i in range(0, x.size(0), x_chunk_size):
                L1_dis[i : i + x_chunk_size, :] = -torch.sum(torch.abs(x[i : i + x_chunk_size, :].unsqueeze(1) - y), -1)
            return L1_dis
        else:
            return -torch.sum(torch.abs(x.unsqueeze(1)-y), -1)

class SkipLSTM(nn.Module):
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0, bidirectional=True, compare=L1()):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)
        self.compare = compare
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
        model = SkipLSTM(21, 50, 1024, 3)
        model_dict = model.state_dict()
        new_dict = {}
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        for key, value in state_dict.items():
            if 'embedding.' in key:
               key = key.replace('embedding.', '')
            if key in model_dict:
                new_dict[key] = value
            
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
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

    def forward(self, x, length):
        batch_size = x.size()[0] // 2
        one_hot = self.to_one_hot(x)
        x_packed = pack_padded_sequence(one_hot, length, batch_first=True, enforce_sorted=False)
        
        hs = []
        for f in self.layers:
            output, (hidden, cell) = f(x_packed)
            output_unpacked, output_length = pad_packed_sequence(output, batch_first=True)
            hs.append(output_unpacked)
            x_packed = output
        emb = torch.mean(torch.cat(hs, dim=2), dim=1)
        emb = self.linear(emb)

        return 10 * (1 + self.similarity(emb[:batch_size], emb[batch_size:]))

    def SSA_score(self, x, length):
        batch_size = x.size()[0] // 2
        one_hot = self.to_one_hot(x)
        x_packed = pack_padded_sequence(one_hot, length, batch_first=True, enforce_sorted=False)
        
        hs = []
        for f in self.layers:
            output, (hidden, cell) = f(x_packed)
            output_unpacked, output_length = pad_packed_sequence(output, batch_first=True)
            hs.append(output_unpacked)
            x_packed = output
        emb = torch.cat(hs, dim=2)
        
        logits = []
        for i in range(batch_size):
            len1, len2 = length[i], length[i + batch_size]
            x1, x2 = emb[i][:len1], emb[i + batch_size][:len2]

            s = self.compare(x1, x2)
            a = torch.softmax(s, 1)
            b = torch.softmax(s, 0)
            
            a = a + b - a*b
            a = a/torch.sum(a)
            
            a = a.view(-1,1)
            s = s.view(-1,1)
            
            logits.append(torch.sum(a*s))
        logits = torch.stack(logits, dim=0)
        # log_p = F.logsigmoid(logits)
        # log_m_p = F.logsigmoid(-logits)
        # zeros = log_p.new(logits.shape[0], 1).zero_()
        # log_p_ge = torch.cat([zeros, log_p], 1)
        # log_p_lt = torch.cat([log_m_p, zeros], 1)
        # log_p = log_p_ge + log_p_lt
        # #print(log_p_ge, log_p_lt, log_p)
        # p = F.softmax(log_p, 1)
        # #print(p)
        # levels = torch.arange(5).to(p.device).float()
        # y_hat = torch.sum(p * levels, 1)
        
        return logits