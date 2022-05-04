from __future__ import print_function,division
import time
from sklearn.decomposition import PCA
import torch
import sys
from typing import List
import numpy as np
import torch.nn.functional as F
from src.prose.alphabets import Uniprot21
BIG_DIST = 1e29

def embed_sequence(model, x, pool='none', use_cuda=False):
    if len(x) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    x = x.cuda()

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        
        # Original : z = model.transform(x) 6165-dim #####
        z = model.transform(x) # 100-dim                           #
        ##################################################

        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        
        z = z.detach().cpu()

    return z

def old_score(logits):
    p = F.sigmoid(logits)
    p_ge = p.new_ones((p.shape[0], p.shape[1] + 1))
    p_ge[:,1:] = torch.log(p)
    p_lt = p.new_ones((p.shape[0], p.shape[1] + 1))
    p_lt[:,:-1] = torch.log(1 - p)
    p = p_ge * p_lt
    print(p_ge, p_lt, p)
    p = p / p.sum() # make sure p is normalized
    levels = torch.arange(5).to(p.device).float()
    y_hat = 4 - torch.sum(p * levels, 1)
    return y_hat

def PCA_svd(X, k, center=True):
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)

    U, S, V = torch.svd(torch.t(X))
    return torch.mm(X, U[:, :, k])

def new_score(logits):
    # logits will be in shape (batch_size, 4)
    #print(logits)
    log_p = F.logsigmoid(logits)
    log_m_p = F.logsigmoid(-logits)
    zeros = log_p.new(logits.shape[0], 1).zero_()
    log_p_ge = torch.cat([zeros, log_p], 1)
    log_p_lt = torch.cat([log_m_p, zeros], 1)
    log_p = log_p_ge + log_p_lt
    #print(log_p_ge, log_p_lt, log_p)
    p = F.softmax(log_p, 1)
    #print(p)
    levels = torch.arange(5).to(p.device).float()
    y_hat = torch.sum(p * levels, 1)

    #print(f"4 - y_hat = {4 - y_hat}")
    return 4 - y_hat

def SSA_score(seqs : List[str], model=None) -> np.ndarray:
    
    assert model is not None

    model.eval()
    model = model.cuda()

    embeddings, length = [], []
    for seq in seqs:
        embedding = embed_sequence(model, seq.encode('utf-8').upper(), pool=None, use_cuda=True)
        length.append(embedding.shape[0])
        embeddings.append(embedding)
    
    # X = torch.cat(embeddings, 0).numpy()
    # pca = PCA(n_components=1000)
    # pca.fit(X)
    # X = pca.transform(X)
    # reduced_embeddings, i = [], 0
    # for leng in length:
    #     reduced_embeddings.append(torch.tensor(X[i : i + leng]))
    #     i += leng


    nodeNum = len(seqs)
    Matrix = np.full((nodeNum, nodeNum), BIG_DIST)
    with torch.no_grad():
        for i in range(len(seqs)):
            for j in range(i):
                # soft_similarity = model.score(embeddings[i].cuda(), embeddings[j].cuda())
                score = -model.score(embeddings[i].cuda(), embeddings[j].cuda())
                # score = new_score(soft_similarity.unsqueeze(0))
                score = score.cpu().detach().numpy()
                Matrix[i][j] = Matrix[j][i] = score
    return Matrix

def prose_embedding(
    seqs : List[str], 
    model=None,
    pool='avg', 
    toks_per_batch=4096
):
    assert model is not None
    
    model.eval()
    model = model.cuda()

    embeddings = []
    for seq in seqs:
        embedding = embed_sequence(model, seq.encode('utf-8').upper(), pool=pool, use_cuda=True)
        embeddings.append(embedding)
    return embeddings
