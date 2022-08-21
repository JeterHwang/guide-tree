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
    def __init__(
        self, 
        nin, 
        nout, 
        hidden_dim, 
        num_layers, 
        dropout=0.3, 
        bidirectional=True, 
        compare=L1(), 
        score_type='SSA',
        RCNN_num=3
    ):
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
        n = 2 * hidden_dim
        # dim = nin
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

        # n = hidden_dim * num_layers + nin
        # if bidirectional:
        #     n = 2 * hidden_dim * num_layers + nin

        ## SSA Embedding
        self.projector = nn.Linear(n, 64)
        
        ## RCNN Embedding
        seq_len = [510, 253, 124]
        blocks = []
        feature_dim = [64, 128, 256, 512]
        for i in range(RCNN_num):
            max_pool_ks = (3 if i == RCNN_num - 1 else 2)
            blocks.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv1d(feature_dim[i], feature_dim[i+1], 3)),
                ('layerNorm', nn.LayerNorm([feature_dim[i+1], seq_len[i]])),
                ('act', nn.LeakyReLU(negative_slope=0.3)),
                ('pool', nn.MaxPool1d(max_pool_ks)),
            ])))
            # blocks.append(nn.GRU(
            #     input_size=50,
            #     hidden_size=50,
            #     num_layers=1,
            #     batch_first=True,
            #     bidirectional=True,
            # ))
            # feature_dim = 2 * 50
        self.RCNN = nn.ModuleList(blocks)
        self.final_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(512, 512, kernel_size=3)),
            ('LayerNorm', nn.LayerNorm([512, 39])),
            ('act', nn.LeakyReLU(negative_slope=0.3)),
        ]))
        self.flatten = nn.Linear(512, 100)
        
        ## MLP Distance
        self.MLP = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(100, 100)),
            ('act0', nn.LeakyReLU(negative_slope=0.3)),
            ('linear1', nn.Linear(100, 1)),
            ('act1', nn.Sigmoid())
        ]))
        
        ## L1 Distance
        ex = 2 * np.sqrt(2 / np.pi) * nout
        var = 4 * (1 - 2 / np.pi) * nout
        beta_init = ex / np.sqrt(var)
        self.theta = nn.Parameter(torch.ones(1) / np.sqrt(var))
        self.beta = nn.Parameter(torch.zeros(1) + beta_init)
        
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
                print(key)
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
        mask = x.ne(-1)
        x = x * mask
        one_hot = self.to_one_hot(x)
        one_hot = one_hot * mask.unsqueeze(2)
        # x_packed = pack_padded_sequence(one_hot, length, batch_first=True, enforce_sorted=False)
        
        output, (hidden, cell) = self.lstm(one_hot)
        # output_unpacked, output_length = pad_packed_sequence(output, batch_first=True)
        hs = output
        
        # hs = [one_hot]
        # h_ = one_hot
        # for f in self.layers:
        #     h, _ = f(h_)
        #     hs.append(h)
        #     h_ = h
        # hs = torch.cat(hs, 2)

        if self.score_type == 'SSA':
            emb = self.projector(hs)
            # emb = []
            # for i in range(len(output_unpacked)):
            #     proj = self.projector(output_unpacked[i][:output_length[i]])
            #     # proj = self.layer_norm(proj)
            #     emb.append(proj)
        elif self.score_type == 'MLP':
            hs = self.projector(hs)
            hs = F.leaky_relu(hs, negative_slope=0.3)
            for block in self.RCNN:
                if isinstance(block, nn.GRU):
                    output, _ = block(hs)
                    residual = torch.cat([hs, hs], dim=2)
                    hs = output + residual
                else:
                    hs = hs.transpose(1, 2).contiguous()
                    hs = block(hs).transpose(1, 2).contiguous()
            hs = self.final_conv(hs.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            hs = self.flatten(hs)
            emb = torch.mean(hs, dim=1)
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
        logits = torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))
        return 1 - logits

    def SSA_score(self, x1, x2):
        logits = []
        for i in range(len(x1)):
            s = torch.cdist(x1[i], x2[i], p=1.0)
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