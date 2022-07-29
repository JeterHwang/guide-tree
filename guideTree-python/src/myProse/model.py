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
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0.2, bidirectional=True, compare=L1()):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(num_layers - 1)])

        self.compare = compare
        self.layers = nn.ModuleList()
        dim = nin
        self.lstm = nn.LSTM(
            nin,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # for i in range(num_layers):
        #     f = nn.LSTM(
        #         dim, 
        #         hidden_dim, 
        #         1, 
        #         batch_first=True, 
        #         bidirectional=bidirectional
        #     )
        #     self.layers.append(f)
        #     if bidirectional:
        #         dim = 2*hidden_dim
        #     else:
        #         dim = hidden_dim

        # n = hidden_dim*num_layers
        # if bidirectional:
        #     n = 2*hidden_dim * 2

        self.classifier = nn.Linear(2 * hidden_dim * 2, 10)
        self.projector = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(2 * hidden_dim, nout)),
            # ("relu", nn.ReLU()),
            # ("linear2", nn.Linear(256, nout))
        ]))
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    @staticmethod
    def load_pretrained(path='prose_dlm'):
        model = SkipLSTM(21, 64, 1024, 3)
        model_dict = model.state_dict()        
        new_dict = {}
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        for key, value in state_dict.items():
            if 'embedding.' in key:
               key = key.replace('embedding.', '')
            if 'layers' in key:
                layer_id = key.split('.')[1]
                postfix = key.split('.')[2].split('_')
                postfix[2] = postfix[2][0] + layer_id
                postfix_revised = '_'.join(postfix)
                key = f"lstm.{postfix_revised}"
            if key in model_dict:
                new_dict[key] = value
            
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        return model

    @staticmethod
    def from_pretrained(path='prose_dlm'):
        model = SkipLSTM(21, 64, 1024, 3)
        state_dict = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
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

    def forward(self, x, length):
        one_hot = self.to_one_hot(x)
        x_packed = pack_padded_sequence(one_hot, length, batch_first=True, enforce_sorted=False)
        
        output, (hidden, cell) = self.lstm(x_packed)
        output_unpacked, output_length = pad_packed_sequence(output, batch_first=True)
        hs = output_unpacked

        # emb1 = torch.mean(hs1, dim=1)
        # emb1 = torch.cat([emb1[:batch_size], emb1[batch_size:]], dim=1)
        # emb1 = self.classifier(emb1)
        # logits1 = emb1

        emb = torch.mean(hs, dim=1)
        emb = self.projector(emb)
        return emb

    def score(self, x, length):
        batch_size = x.size()[0] // 2
        emb = self.forward(x, length) 
        logits = torch.exp(-torch.sum(torch.abs(emb[:batch_size] - emb[batch_size:]), dim=1))
        return 1 - logits

    def SSA_score(self, x, length):
        batch_size = x.size()[0] // 2
        one_hot = self.to_one_hot(x)
        x_packed = pack_padded_sequence(one_hot, length, batch_first=True, enforce_sorted=False)
        
        # hs = []
        # for f in self.layers:
        #     output, (hidden, cell) = f(x_packed)
        #     output_unpacked, output_length = pad_packed_sequence(output, batch_first=True)
        #     hs.append(output_unpacked)
        #     x_packed = output
        output, (hidden, cell) = self.lstm(x_packed)
        output_unpacked, output_length = pad_packed_sequence(output, batch_first=True)
        emb = output_unpacked
        
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