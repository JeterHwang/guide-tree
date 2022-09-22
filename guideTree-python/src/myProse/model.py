import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
import math
import logging
import numpy as np

class L1(nn.Module):
    def forward(self, x, y, chunk_size=40000):
        if x.size(1) > 100:
            x_chunk_size = chunk_size // y.size(0) + 1
            L1_dis = torch.zeros(x.size(0), y.size(0), device=x.device)
            for i in range(0, x.size(0), x_chunk_size):
                L1_dis[i : i + x_chunk_size, :] = torch.sum(torch.abs(x[i : i + x_chunk_size, :].unsqueeze(1) - y), -1)
            return L1_dis
        else:
            return torch.sum(torch.abs(x.unsqueeze(1)-y), -1)

class SkipLSTM(nn.Module):
    def __init__(
        self, 
        nin, 
        nout, 
        hidden_dim, 
        num_layers, 
        dropout=0.3, 
        bidirectional=True, 
        compare=L1(), 
        score_type='MLP',
        esm_model=None,
        RCNN_num=3
    ):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(num_layers - 1)])

        self.compare = compare
        self.score_type = score_type

        if esm_model is not None:
            self.esm = esm_model
            self.repr_layers = num_layers
            n = hidden_dim
        else:
            # self.lstm = nn.LSTM(
            #     nin,
            #     hidden_dim,
            #     num_layers,
            #     batch_first=True,
            #     bidirectional=bidirectional
            # )
            dim = nin
            self.lstm = nn.ModuleList()
            for i in range(num_layers):
                f = nn.LSTM(
                    dim, 
                    hidden_dim, 
                    1, 
                    batch_first=True, 
                    bidirectional=bidirectional
                )
                self.lstm.append(f)
                dim = 2 * hidden_dim
            n = 2 * hidden_dim * 3 + nin
        self.proj = nn.Linear(n, nout)
        
    @staticmethod
    def load_pretrained(path='prose_dlm', score_type='SSA', esm_model=None, layers=3, hidden_dim=1024, output_dim=100):
        model = SkipLSTM(
            21, 
            output_dim, 
            hidden_dim, 
            layers, 
            score_type=score_type, 
            esm_model=esm_model
        )
        model_dict = model.state_dict()        
        new_dict = {}
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        for key, value in state_dict.items():
            if 'embedding.' in key:
                key = key.replace('embedding.', '')
            if 'layers' in key:
                key = key.replace('layers', 'lstm')
            # if 'layers' in key:
            #     layer_id = key.split('.')[1]
            #     postfix = key.split('.')[2].split('_')
            #     postfix[2] = postfix[2][0] + layer_id
            #     postfix_revised = '_'.join(postfix)
            #     key = f"lstm.{postfix_revised}"
            if key in model_dict:
                logging.info(key)
                new_dict[key] = value
            
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        return model

    @staticmethod
    def from_pretrained(path='prose_dlm', esm_model=None):
        model = SkipLSTM(21, 100, 1024, 3, esm_model=esm_model)
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
        if hasattr(self, 'lstm'):
            one_hot = self.to_one_hot(x)
            # h_ = pack_padded_sequence(one_hot, length, batch_first=True, enforce_sorted=False)
            # output, (hidden, cell) = self.lstm(h_)
            # output_unpacked, _ = pad_packed_sequence(output, batch_first=True)
            # hs = output_unpacked
            hs = []
            h_ = pack_padded_sequence(one_hot, length, batch_first=True, enforce_sorted=False)
            for f in self.lstm:
                h, _ = f(h_)
                h_ = h
            h_unpacked, _ = pad_packed_sequence(h_, batch_first=True)
            # hs.append(h_unpacked)
            # hs = torch.cat(hs, dim=2)
            # hs = self.proj(hs)
            hs = h_unpacked
        else:
            results = self.esm(x, repr_layers=[self.repr_layers], return_contacts=False)
            hs = results["representations"][self.repr_layers][:,1:,:]
            # bos = results["representations"][self.repr_layers][:,0,:]

        if self.score_type == 'SSA':
            hs = self.proj(hs)
            emb = []
            for i in range(x.size(0)):
                emb.append(hs[i][:length[i]])
        elif self.score_type == 'MLP':
            pooling = []
            for i in range(x.size(0)):
                pooling.append(torch.mean(hs[i][:length[i]], dim=0))
            emb = torch.stack(pooling, dim=0)
        else:
            emb = torch.mean(hs, dim=1)
            emb = self.projector(emb)
        return emb
    
    def score(self, emb):
        batch_size = len(emb) // 2
        if self.score_type == 'SSA':
            return self.SSA_score(emb[:batch_size], emb[batch_size:])
        elif self.score_type == 'MLP':
            return self.MLP_score(emb[:batch_size], emb[batch_size:])
        else:
            return self.L1_score(emb[:batch_size], emb[batch_size:])

    def MLP_score(self, x1, x2):
        # mask = F.softmax(x1 * x2, dim=1)
        # logits = torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))
        logits = torch.sum(torch.abs(x1 - x2), dim=1)
        return logits

    def SSA_score(self, x1, x2):
        s = torch.cdist(x1, x2, 1)
        a = torch.softmax(s, dim=2)
        b = torch.softmax(s, dim=1)
        a = a + b - a * b
        
        a = a / torch.sum(a, dim=[1,2], keepdim=True)
        a = a.view(a.size(0), -1, 1)
        s = s.view(s.size(0), -1, 1)
        logits = torch.sum(a * s, dim=[1,2])
        return logits

    def L1_score(self, x1, x2):
        logits = torch.exp(-self.theta * torch.sum(torch.abs(x1 - x2), dim=1) - self.beta)
        return 1 - logits

    def get_parameters(self, keys=None, mode='include'):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad: yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad: yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad: yield param
        else:
            raise ValueError('do not support: %s' % mode)