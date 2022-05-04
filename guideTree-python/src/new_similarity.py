from __future__ import print_function, division
from math import sqrt

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm

from src.alphabets import Uniprot21
from src.dataset import SSADataset
BIG_DIST = 1e29

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
    batch_size=1, 
    device=0, 
    coarse=False
):
    
    ## load the data
    alphabet = Uniprot21()
    dataset = SSADataset(seq1, seq2, alphabet)
    test_queue = torch.utils.data.DataLoader(
        dataset,
        # batch_sampler=dataset.get_batch_indices(4096),
        batch_size=16,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        #num_workers=8
    )
    ## load the model    
    model = torch.load(model_path)
    model = model.cuda()
    model.eval()

    Matrix = np.full((nodeNum, nodeNum), BIG_DIST)
    scores = []
    with torch.no_grad():
        for data, order in tqdm(test_queue):
            data = data.cuda()
            output = model(data) # embed the sequences
            embedding, seq_length = pad_packed_sequence(output, batch_first=True)
            # Reordering 
            reorder_embedding = [None] * embedding.shape[0]
            for i in range(len(order)):
                #print(embedding[i, : seq_length[i]])
                reorder_embedding[order[i]] = embedding[i, : seq_length[i]]
            
            assert embedding.shape[0] % 2 == 0
            n = embedding.shape[0] // 2
            #print(reorder_embedding)
            for i in range(n):
                seq1_embed = reorder_embedding[i]
                seq2_embed = reorder_embedding[i + n]

                logits = model.score(seq1_embed, seq2_embed)
                p = torch.sigmoid(logits).cpu()
                p_ge = torch.ones(p.size(0)+1)
                p_ge[1:] = p
                p_lt = torch.ones(p.size(0)+1)
                p_lt[:-1] = 1 - p
                p = p_ge*p_lt
                p = p/p.sum() # make sure p is normalized
                levels = torch.arange(5).float()
                score = 4 - torch.sum(p*levels).item()
                scores.append(score)
    #print(scores[:5])
    k = 0
    for i in range(nodeNum):
        for j in range(i):
            Matrix[i][j] = Matrix[j][i] = scores[k]
            k = k + 1
    return Matrix

        
        
