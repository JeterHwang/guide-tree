import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
import math
import numpy as np

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
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0.3, bidirectional=True, compare=L1(), score_type='SSA'):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(num_layers - 1)])

        self.compare = compare
        self.score_type = score_type
        self.layers = nn.ModuleList()
        self.lstm = nn.LSTM(
            nin,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
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

        # self.classifier = nn.Linear(2 * hidden_dim * 2, 10)
        self.projector = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(2 * hidden_dim, nout)),
            # ("relu", nn.ReLU()),
            # ("linear2", nn.Linear(256, nout))
        ]))
        # self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.query_proj = nn.Linear(nout, nout)
        # self.key_proj = nn.Linear(nout, nout)
        ex = 2 * np.sqrt(2 / np.pi) * nout
        var = 4 * (1 - 2 / np.pi) * nout
        beta_init = ex / np.sqrt(var)
        self.theta = nn.Parameter(torch.ones(1) / np.sqrt(var))
        self.beta = nn.Parameter(torch.zeros(1) + beta_init)
        self.layer_norm = nn.LayerNorm(nout)

    @staticmethod
    def load_pretrained(path='prose_dlm', score_type='SSA'):
        model = SkipLSTM(21, 64, 1024, 3, score_type=score_type)
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
        if self.score_type == 'SSA':
            emb = []
            for i in range(len(output_unpacked)):
                proj = self.projector(output_unpacked[i][:output_length[i]])
                # proj = self.layer_norm(proj)
                emb.append(proj)
        else:
            emb = torch.mean(hs, dim=1)
            # emb = self.projector(emb)
        # attn_output, attn_output_weights = self.attn(emb, emb, emb)
        # print(torch.sum(attn_output_weights, dim=2))
        return emb
    
    def score(self, emb):
        batch_size = len(emb) // 2
        if self.score_type == 'SSA':
            return self.SSA_score(emb[:batch_size], emb[batch_size:])
        else:
            return self.L1_score(emb[:batch_size], emb[batch_size:])

    def SSA_score(self, x1, x2):
        logits = []
        for i in range(len(x1)):
            s = torch.cdist(x1[i], x2[i], p=1.0)
            
            # qa = self.query_proj(emb[i])
            # qb = self.query_proj(emb[i + batch_size]).transpose(0,1).contiguous()
            # ka = self.key_proj(emb[i])
            # kb = self.key_proj(emb[i + batch_size]).transpose(0,1).contiguous()
            # mat1 = torch.matmul(qa, kb) / math.sqrt(self.nout)
            # mat2 = torch.matmul(ka, qb) / math.sqrt(self.nout)

            a = torch.softmax(s, dim=1)
            b = torch.softmax(s, dim=0)
            a = a + b - a * b
            a = a / torch.sum(a)
            a = a.view(-1, 1)
            s = s.view(-1, 1)
            dist = torch.sum(a * s)
            logits.append(1 - torch.exp(-dist))

        logits = torch.stack(logits, dim=0).view(-1)
        return logits

    def L1_score(self, x1, x2):
        logits = torch.exp(-self.theta * torch.sum(torch.abs(x1 - x2), dim=1) - self.beta)
        return 1 - logits