import json
import os
from typing import Mapping
from weakref import finalize
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
import time 
import copy
import torch
import numpy as np
import subprocess
from subprocess import PIPE
from scipy.stats import pearsonr, spearmanr
from gmm import GMM_Batch
from kmeans_pytorch import kmeans
import logging
BIG_DIST = 1000000000

def calculate_corr(ground, predicted):
    spearman, _ = spearmanr(np.array(ground), np.array(predicted))
    pearson, _ = pearsonr(np.array(ground), np.array(predicted))
    return spearman, pearson

def runcmd(command):
    bash_command = command.split()
    ret = subprocess.run(bash_command, stdout=PIPE, stderr=PIPE)
    if ret.returncode == 0:
        return ret.stdout
    else:
        logging.error(f"Error : {ret.stderr}")
        return ret.stderr

def read_data(seq_path : Path, step=3):
    pairs = []
    SCOPe = {}
    with open(seq_path, 'r') as f:
        for line in tqdm(f.readlines(), desc=f"Reading {seq_path.name}"):
            line = json.loads(line)
            seqA, seqB = line['A'], line['B']
            if len(seqA) <= 510 and len(seqB) <= 510:
                if seqA not in SCOPe:
                    SCOPe[seqA] = [line]
                else:
                    SCOPe[seqA].append(line)
    for seqA, lines in SCOPe.items():
        for i in range(0, len(lines), 3):
            pairs.append(lines[i])
    return pairs

def L1_exp_distance(x, chunk_size=40000):
    x = x.cuda()
    # print(x.size())
    with torch.no_grad():
        if x.size(0) > 10000:
            x_chunk_size = chunk_size // x.size(0) + 1
            L1_dis = torch.zeros(x.size(0), x.size(0))
            # print('Successfully Allocate Memory !!')
            for i in tqdm(range(0, x.size(0), x_chunk_size), desc="Building DIS Matrix:"):
                L1_dis[i : i + x_chunk_size, :].data.copy_(torch.sum(torch.abs(x[i : i + x_chunk_size, :].unsqueeze(1) - x), -1).data)
            L1_dis.fill_diagonal_(BIG_DIST)
        else:
            L1_dis = torch.sum(torch.abs(x.unsqueeze(1)-x), -1)
            L1_dis.fill_diagonal_(BIG_DIST)
    return L1_dis.cpu().detach().numpy()
        
def SSA_score_fast(x1, x2):
    s = torch.cdist(x1, x2, 1)
    a = torch.softmax(s, dim=2)
    b = torch.softmax(s, dim=1)
    a = a + b - a * b  
    a = a / torch.sum(a, dim=[1,2], keepdim=True)
    a = a.view(a.size(0), -1, 1)
    s = s.view(s.size(0), -1, 1)
    logits = torch.sum(a * s, dim=[1,2])
    return logits

def SSA_score_slow(x1, x2):
    assert len(x1) == len(x2)
    logits = []
    for i in range(len(x1)):
        x1[i] = x1[i].cuda()
        x2[i] = x2[i].cuda()
        s = torch.sum(torch.abs(x1[i].unsqueeze(1)-x2[i]), -1)
        a = torch.softmax(s, dim=1)
        b = torch.softmax(s, dim=0)
        a = a + b - a * b  
        a = a / torch.sum(a)
        a = a.view(-1, 1)
        s = s.view(-1, 1)
        logits.append(torch.sum(a*s))
    return logits

def L2_distance(x, chunk_size=40000):
    x = x.cuda()
    with torch.no_grad():
        if x.size(0) > 10000:
            x_chunk_size = chunk_size // x.size(0) + 1
            L2_dis = torch.zeros(x.size(0), x.size(0))
            for i in tqdm(range(0, x.size(0), x_chunk_size), desc="Building DIS Matrix:"):
                L2_dis[i : i + x_chunk_size, :].data.copy_(torch.sum((x[i : i + x_chunk_size, :].unsqueeze(1) - x)**2, -1).data)
            L2_dis.fill_diagonal_(BIG_DIST)
        else:
            L2_dis = torch.sum((x.unsqueeze(1)-x)**2, -1)
            L2_dis.fill_diagonal_(BIG_DIST)
    return L2_dis.cpu().detach().numpy()

def UPGMA(distmat, seqID, tree_dir):
    leafNode = len(seqID)
    internalNode = len(seqID) - 1
    newID = len(seqID) - 1

    matrix2id = np.arange(leafNode)
    height = np.zeros(leafNode + internalNode)
    parent = np.ones(leafNode + internalNode).astype(int) * (-1)
    Llength = np.zeros(leafNode + internalNode)
    Rlength = np.zeros(leafNode + internalNode)
    Lindex = np.ones(leafNode + internalNode).astype(int) * (-1)
    Rindex = np.ones(leafNode + internalNode).astype(int) * (-1)
    RNN, NN = [], []
    
    ## UPGMA
    for iter in range(internalNode):
        if len(RNN) == 0:
            for i in range(leafNode):
                if matrix2id[i] != -1:
                    cand_MatrixID = i
                    break
            RNN.append(cand_MatrixID)
            cand_NN_MatrixID = np.argmin(distmat[cand_MatrixID])
            NN.append(cand_NN_MatrixID)

        while len(RNN) == 1 or not (NN[-2] == RNN[-1] and NN[-1] == RNN[-2]):
            cand_MatrixID = NN[-1]
            RNN.append(cand_MatrixID)
            cand_NN_MatrixID = np.argmin(distmat[cand_MatrixID])
            NN.append(cand_NN_MatrixID)
        
        Lmin = RNN[-1]
        Rmin = RNN[-2]
        newID += 1
        dLR = distmat[Lmin][Rmin]
        new_height = dLR / 2
        height[newID] = new_height
        left_child_index = matrix2id[Lmin]
        right_child_index = matrix2id[Rmin]
        Lindex[newID] = left_child_index
        Rindex[newID] = right_child_index
        Llength[newID] = new_height - height[left_child_index]
        Rlength[newID] = new_height - height[right_child_index]
        parent[left_child_index] = newID
        parent[right_child_index] = newID

        matrix2id[Lmin] = newID
        matrix2id[Rmin] = -1
        for i in range(leafNode):
            if matrix2id[i] == -1:
                continue
            newDist = (distmat[i][Lmin] + distmat[i][Rmin]) / 2
            if i != Lmin:
                distmat[Lmin][i] = distmat[i][Lmin] = newDist
            distmat[i][Rmin] = distmat[Rmin][i] = BIG_DIST
        RNN = RNN[:-2]
        NN = NN[:-2]
        if len(RNN) != 0:
            NN[-1] = np.argmin(distmat[RNN[-1]])
    
    ## Write Guide Tree to File
    visited = np.zeros(leafNode + internalNode)
    root = newID
    with open(tree_dir, 'w') as f:
        while True:
            if (Lindex[root] != -1 and Lindex[root] >= root) or (Rindex[root] != -1 and Rindex[root] >= root):
                logging.error('Loop Found !!')
            
            visited[root] = 1
            if root < leafNode:
                parent_idx = int(parent[root])
                if Lindex[parent_idx] == root:
                    f.write(f"{seqID[root]}:{Llength[parent_idx]}\n")
                elif Rindex[parent_idx] == root:
                    f.write(f"{seqID[root]}:{Rlength[parent_idx]}\n")
                else:
                    logging.error("ERR !!")
                root = parent_idx
            elif visited[int(Lindex[root])] == 0:
                f.write('(\n')
                root = int(Lindex[root])
            elif visited[int(Rindex[root])] == 0:
                f.write(',\n')
                root = int(Rindex[root])
            else:
                if parent[root] == -1:
                    f.write(f")\n")
                    break
                else:
                    parent_idx = int(parent[root])
                    if Lindex[parent_idx] == root:
                        f.write(f"):{Llength[parent_idx]}\n")
                    elif Rindex[parent_idx] == root:
                        f.write(f"):{Rlength[parent_idx]}\n")
                    else:
                        logging.error("ERR !!")
                    root = parent_idx
        f.write(';')

def BisectingKmeans(seqs, min_cluster_size=500):
    device = torch.cuda.current_device()
    start_time = time.time()
    # Exclude identical sequences
    final_cluster = [{
        'center' : None,
        'seqs' : copy.deepcopy(seqs)
    }]
    additional_cluster = []
    while len(final_cluster[0]['seqs']) > min_cluster_size:
        # Extract MAX
        biggest_cluster = final_cluster[0]
        clusterPoints = biggest_cluster['seqs']
        # POP
        final_cluster = final_cluster[1:]
        
        # make coordinate array
        x = []
        for seq in clusterPoints:
            x.append(seq['embedding'])
        x = torch.stack(x, dim=0)
        
        cluster_num = 2
        cluster_ids, cluster_centers = kmeans(
            X = x,
            num_clusters = cluster_num,
            distance = 'euclidean',
            device = device,
            tqdm_flag=False,
        )
        
        newCluster = [[] for _ in range(cluster_num)]
        assert len(cluster_ids) == len(clusterPoints)
        
        for clusterID, seq in zip(cluster_ids, clusterPoints):
            newCluster[clusterID].append(seq)
        for center, seq in zip(cluster_centers, newCluster):
            if len(seq) == 0:
                continue
            if len(seq) == len(clusterPoints): ## Cannot divide anymore !!
                additional_cluster.append({
                    'center' : center,
                    'seqs' : seq
                })
            else:
                final_cluster.append({
                    'center' : center,
                    'seqs' : seq
                })
        final_cluster = sorted(final_cluster, key=lambda x : len(x['seqs']), reverse=True)
    
    final_cluster = final_cluster + additional_cluster
    # if len(final_cluster) > 1:
    #     clusterPoints = copy.deepcopy(seqs)
    #     cluster_num = len(final_cluster)
    #     centers = torch.stack([cluster['center'] for cluster in final_cluster])
    #     x = torch.stack([point['embedding'] for point in clusterPoints])
    #     cluster_ids, cluster_centers = kmeans(
    #         X = x,
    #         num_clusters = len(final_cluster),
    #         cluster_centers = centers,
    #         distance = 'euclidean',
    #         device = device,
    #         tqdm_flag=False,
    #     )
    #     newCluster = [[] for _ in range(cluster_num)]
    #     assert len(cluster_ids) == len(clusterPoints)
        
    #     for clusterID, seq in zip(cluster_ids, clusterPoints):
    #         newCluster[clusterID].append(seq)
    #     final_cluster = []
    #     for center, seq in zip(cluster_centers, newCluster):
    #         if len(seq) == 0:
    #             continue
    #         final_cluster.append({
    #             'center' : center,
    #             'seqs' : seq
    #         })  

    final_cluster.sort(key=lambda x : len(x['seqs']), reverse=True)
    
    # Calculate Loss
    loss = 0
    for cluster in final_cluster:
        cent, sqs = cluster['center'], cluster['seqs']
        if cent is None:
            continue
        for seq in sqs:
            loss += torch.sum(torch.abs(cent - seq['embedding']))
    logging.info(f'Average K-means++ Loss : {loss / len(seqs)} !!')
    
    centers, clusters = [], []
    for clusterID, ele in enumerate(final_cluster):
        center = ele['center']
        cluster_seqs = ele['seqs']
        
        assert len(cluster_seqs) != 0
        
        centers.append({
            'name' : f"precluster-{clusterID}",
            'embedding' : center
        })
        clusters.append(cluster_seqs)
    logging.info(f'Finish K-means in {time.time() - start_time} seconds')
    return centers, clusters

def identical_seqs_to_subtree(identical_seqs):
    lines = []
    for i, seq in enumerate(identical_seqs):
        if i == len(identical_seqs) - 1:
            lines.append(f"\n{seq['name']}\n")
            lines.append(f"{':0.00000)' * (len(identical_seqs) - 1)}")
        else:
            lines.append('(\n')
            lines.append(f"{seq['name']}\n")
            lines.append(f":0.00000,")
    return lines

def UPGMA_Kmeans(distmat, clusters, id2cluster, tree_path, fasta_dir, dist_type="NW"):
    ## Use MAFFT to build guide tree in sub-clusters
    tree_files = []
    for i, cluster in enumerate(tqdm(clusters, desc="Construct Subtrees:")):
        name_mapping = {}
        fasta_path = fasta_dir / f"{tree_path.stem}-{i}.fa"
        with open(fasta_path, 'w') as fa:
            if len(cluster) == 0:
                logging.error('Error : Empty Subcluster !!')
            for seq in cluster:
                name_mapping[seq['name'].replace('.', '_').replace('|', '_')] = seq['name']
                fa.write(f">{seq['name']}\n")
                fa.write(f"{seq['seq']}\n")
        
        if len(cluster) == 1:
            if id2cluster[cluster[0]['name']] is not None:
                lines = ["\n"] + identical_seqs_to_subtree(id2cluster[cluster[0]['name']]) + [";"]
            else:
                lines = [f"\n{cluster[0]['name']}\n", ";"]
            tree_files.append(lines)
            continue
        
        sub_tree_path = fasta_dir / f"{fasta_path.name}.tree"
        if dist_type == "NW":
            runcmd(f"./mafft --globalpair --anysymbol --thread 16 --treeout {fasta_path.absolute().resolve()}")
        elif dist_type == "SW":
            runcmd(f"./mafft --localpair --anysymbol --thread 16 --treeout {fasta_path.absolute().resolve()}")
        elif dist_type == "LCS":
            runcmd(f"famsa -gt upgma -t 8 -gt_export {fasta_path.absolute().resolve()} {sub_tree_path.absolute().resolve()}")
        else:
            raise NotImplementedError
        lines = []
        with open(sub_tree_path, 'r') as tree:
            for line in tree:
                if dist_type in ['NW', 'SW']:
                    underscore = line.find('_')
                    if underscore != -1:
                        line = line[underscore+1:]
                    seq_name = line.replace('\n', '')
                    if seq_name in name_mapping:
                        seq_name = name_mapping[seq_name]
                        if id2cluster[seq_name] is not None:
                            lines[-1] = lines[-1].replace('\n', '')
                            line = identical_seqs_to_subtree(id2cluster[seq_name])
                        else:
                            line = seq_name + '\n'
                if isinstance(line, str):
                    lines.append(line)
                else:
                    lines = lines + line
        tree_files.append(lines)
    
    if len(tree_files) == 1:
        with open(tree_path, 'w') as f:
            for line in tree_files[0]:
                # print(line[-4:])
                f.write(line)
        return

    leafNode = len(distmat)
    internalNode = len(distmat) - 1
    newID = len(distmat) - 1

    matrix2id = np.arange(leafNode)
    height = np.zeros(leafNode + internalNode)
    parent = np.ones(leafNode + internalNode).astype(int) * (-1)
    Llength = np.zeros(leafNode + internalNode)
    Rlength = np.zeros(leafNode + internalNode)
    Lindex = np.ones(leafNode + internalNode).astype(int) * (-1)
    Rindex = np.ones(leafNode + internalNode).astype(int) * (-1)
    RNN, NN = [], []
    
    ## UPGMA
    for iter in range(internalNode):
        if len(RNN) == 0:
            for i in range(leafNode):
                if matrix2id[i] != -1:
                    cand_MatrixID = i
                    break
            RNN.append(cand_MatrixID)
            cand_NN_MatrixID = np.argmin(distmat[cand_MatrixID])
            NN.append(cand_NN_MatrixID)

        while len(RNN) == 1 or not (NN[-2] == RNN[-1] and NN[-1] == RNN[-2]):
            cand_MatrixID = NN[-1]
            RNN.append(cand_MatrixID)
            cand_NN_MatrixID = np.argmin(distmat[cand_MatrixID])
            NN.append(cand_NN_MatrixID)
        
        Lmin = RNN[-1]
        Rmin = RNN[-2]
        newID += 1
        dLR = distmat[Lmin][Rmin]
        new_height = dLR / 2
        height[newID] = new_height
        left_child_index = matrix2id[Lmin]
        right_child_index = matrix2id[Rmin]
        Lindex[newID] = left_child_index
        Rindex[newID] = right_child_index
        Llength[newID] = new_height - height[left_child_index]
        Rlength[newID] = new_height - height[right_child_index]
        parent[left_child_index] = newID
        parent[right_child_index] = newID

        matrix2id[Lmin] = newID
        matrix2id[Rmin] = -1
        for i in range(leafNode):
            if matrix2id[i] == -1:
                continue
            newDist = (distmat[i][Lmin] + distmat[i][Rmin]) / 2
            if i != Lmin:
                distmat[Lmin][i] = distmat[i][Lmin] = newDist
            distmat[i][Rmin] = distmat[Rmin][i] = BIG_DIST
        RNN = RNN[:-2]
        NN = NN[:-2]
        if len(RNN) != 0:
            NN[-1] = np.argmin(distmat[RNN[-1]])
    
    ## Write Guide Tree to File
    visited = np.zeros(leafNode + internalNode)
    root = newID
    with open(tree_path, 'w') as f:
        while True:
            if (Lindex[root] != -1 and Lindex[root] >= root) or (Rindex[root] != -1 and Rindex[root] >= root):
                logging.error('Loop Found !!')
            
            visited[root] = 1
            if root < leafNode:
                parent_idx = int(parent[root])
                if Lindex[parent_idx] == root:
                    for idx, line in enumerate(tree_files[root]):
                        if idx == len(tree_files[root]) - 1:
                            line = line.replace('\n', '')
                        if ";" in line:
                            line = line.replace(";", f":{Llength[parent_idx]}")
                            # line = line.replace(";", f":1.0")
                        f.write(line)
                elif Rindex[parent_idx] == root:
                    for idx, line in enumerate(tree_files[root]):
                        if idx == len(tree_files[root]) - 1:
                            line = line.replace('\n', '')
                        if ";" in line:
                            line = line.replace(";", f":{Rlength[parent_idx]}")
                            # line = line.replace(";", f":1.0")
                        f.write(line)
                else:
                    logging.error("Tree Construction Error !!")
                root = parent_idx
            elif visited[int(Lindex[root])] == 0:
                f.write('(')
                root = int(Lindex[root])
            elif visited[int(Rindex[root])] == 0:
                f.write(',')
                root = int(Rindex[root])
            else:
                if parent[root] == -1:
                    f.write(f")")
                    break
                else:
                    parent_idx = int(parent[root])
                    if Lindex[parent_idx] == root:
                        f.write(f"):{Llength[parent_idx]}")
                    elif Rindex[parent_idx] == root:
                        f.write(f"):{Rlength[parent_idx]}")
                    else:
                        logging.error("Tree Construction Error !!")
                    root = parent_idx
        f.write(';')