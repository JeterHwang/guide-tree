from __future__ import print_function,division
import time
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import torch
import sys
from typing import Dict, List
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from src.prose.alphabets import Uniprot21
from src.dataset import proseDataset, LSTMDataset
BIG_DIST = 1e29

def embed_sequence(model, x, pool='none'):
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
        z = model.transform(x)                 # 100-dim #
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

def SSA_score(seqs : List[Dict], model=None, save_path=None) -> np.ndarray:
    assert model is not None
    assert save_path is not None
    model = model.cuda()
    model.eval()
        
    # X = torch.cat(embeddings, 0).numpy()
    # pca = PCA(n_components=1000)
    # pca.fit(X)
    # X = pca.transform(X)
    # reduced_embeddings, i = [], 0
    # for leng in length:
    #     reduced_embeddings.append(torch.tensor(X[i : i + leng]))
    #     i += leng


    nodeNum = len(seqs)
    # test_dataset = proseDataset(seqs, save_path)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=8
    # )
    Matrix = np.full((nodeNum, nodeNum), BIG_DIST)
    score_max = 0
    with torch.no_grad(), tqdm(total=nodeNum * (nodeNum - 1) // 2, desc='Scoring Matrix') as t:
        for i in range(nodeNum):
            for j in range(i):
                emb1 = torch.load(save_path / f"{seqs[i]['name'].replace('/', '-')}.pt").cuda()
                emb2 = torch.load(save_path / f"{seqs[j]['name'].replace('/', '-')}.pt").cuda()
                score = -model.score(emb1, emb2)
                # soft_similarity = model.score(embeddings[i].cuda(), embeddings[j].cuda())
                # score = new_score(soft_similarity.unsqueeze(0))
                score = score.cpu().detach().numpy()
                score_max = max(score_max, score)
                Matrix[i][j] = Matrix[i][j] = score
                t.update(1)
        # for batch_idx, (emb1, emb2, x, y) in enumerate(tqdm(test_loader, desc="Scoring Matrix")): 
        #     emb1 = emb1.squeeze(0).cuda()
        #     emb2 = emb2.squeeze(0).cuda()
        #     score = -model.score(emb1, emb2)
        #     # soft_similarity = model.score(embeddings[i].cuda(), embeddings[j].cuda())
        #     # score = new_score(soft_similarity.unsqueeze(0))
        #     score = score.cpu().detach().numpy()
        #     score_max = max(score_max, score)
        #     Matrix[x[0]][y[0]] = Matrix[y[0]][x[0]] = score
            
    return Matrix / score_max

def prose_embedding(
    seqs : List[Dict], 
    model=None,
    pool='avg', 
    toks_per_batch=4096,
    save_path=None
):
    assert model is not None
    
    model.eval()
    model = model.cuda()
    embedding_dataset = LSTMDataset(seqs, Uniprot21())
    embedding_dataloader = torch.utils.data.DataLoader(
        embedding_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    embeddings = []
    with torch.no_grad():
        for name, data in tqdm(embedding_dataloader, desc="LSTM embedding"):
            data = data.long().cuda()
            # Original : z = model.transform(x) 6165-dim #####
            output = model.transform(data) # 100-dim         #
            ##################################################
            embedding = output.squeeze(0)
            torch.save(embedding, save_path / f"{name[0].replace('/', '-')}.pt")
            if pool == 'sum':
                embedding = embedding.sum(0)
            elif pool == 'max':
                embedding, _ = embedding.max(0)
            elif pool == 'avg':
                embedding = embedding.mean(0)
            embedding = embedding.detach().cpu().numpy()
            embeddings.append(embedding)
    return embeddings
