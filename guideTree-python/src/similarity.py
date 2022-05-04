from __future__ import print_function, division
from math import sqrt

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable

from src.alphabets import Uniprot21

BIG_DIST = 1e29

def pack_sequences(X, order=None):
    
    #X = [x.squeeze(0) for x in X]
    
    n = len(X)
    lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]
    m = max(len(x) for x in X)
    
    X_block = X[0].new(n,m).zero_()
    
    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i,:len(x)] = x
        
    #X_block = torch.from_numpy(X_block)
        
    lengths = lengths[order]
    X = pack_padded_sequence(X_block, lengths, batch_first=True)
    
    return X, order


def unpack_sequences(X, order):
    X,lengths = pad_packed_sequence(X, batch_first=True)
    X_block = [None]*len(order)
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i,:lengths[i]]
    return X_block

def encode_sequence(x, alphabet):
    # convert to bytes and uppercase
    x = x.encode('utf-8').upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    return x


def load_pairs(seq1, seq2, alphabet):
    x0 = [encode_sequence(x, alphabet) for x in seq1]
    x1 = [encode_sequence(x, alphabet) for x in seq2]

    return x0, x1

class TorchModel:
    def __init__(self, model, use_cuda, mode='ssa'):
        self.model = model
        self.use_cuda = use_cuda
        self.mode = mode

    def __call__(self, x, y):
        n = len(x)
        c = [torch.from_numpy(x_).long() for x_ in x] + [torch.from_numpy(y_).long() for y_ in y]

        c, order = pack_sequences(c)
        if self.use_cuda:
            c = c.cuda()
        c = PackedSequence(Variable(c.data), c.batch_sizes)
        with torch.no_grad():
            z = self.model(c) # embed the sequences
            z = unpack_sequences(z, order)

            scores = np.zeros(n)
            for i in range(n):
                z_x = z[i]
                z_y = z[i+n]

                logits = self.model.score(z_x, z_y)
                p = torch.sigmoid(logits).cpu()
                p_ge = torch.ones(p.size(0)+1)
                p_ge[1:] = p
                p_lt = torch.ones(p.size(0)+1)
                p_lt[:-1] = 1 - p
                p = p_ge*p_lt
                p = p/p.sum() # make sure p is normalized
                levels = torch.arange(5).float()
                scores[i] = 4 - torch.sum(p*levels).item()

        return scores

def score_pairs(model, x0, x1, batch_size=100):
    scores = []
    for i in range(0, len(x0), batch_size):
        x0_mb = x0[i:i+batch_size]
        x1_mb = x1[i:i+batch_size]
        scores.append(model(x0_mb, x1_mb))
    scores = np.concatenate(scores, 0)
    return scores

def calculate_score(model, z_x, z_y):
    model = model.model
    s = model.compare(z_x, z_y) # L1 distance
    a = F.softmax(s, 1)
    b = F.softmax(s, 0)
    a = a + b - a*b
    c = torch.sum(a*s)/torch.sum(a)
    return -c

def SSA_score(
    seq1, 
    seq2, 
    nodeNum,
    model_path='./ckpt/iclr2019/pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_p0.05_epoch100.sav', 
    batch_size=16, 
    device=0, 
    coarse=False
):
    
    ## load the data
    alphabet = Uniprot21()

    ## load the model    
    print('----- Loadding Pretrained Model -----')
    model = torch.load(model_path)
    model.eval()
    print('----- finish -----')

    ## set the device
    d = device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    if use_cuda:
        model.cuda()

    mode = 'align'
    if coarse:
        mode = 'coarse'
    model = TorchModel(model, use_cuda, mode=mode)

    x0_test, x1_test = load_pairs(seq1, seq2, alphabet)
    scores = score_pairs(model, x0_test, x1_test, batch_size)
    print(scores)
    Matrix = np.full((nodeNum, nodeNum), BIG_DIST)
    k = 0
    for i in range(nodeNum):
        for j in range(i):
            Matrix[i][j] = Matrix[j][i] = scores[k]
            k += 1
    
    return Matrix

        
        
