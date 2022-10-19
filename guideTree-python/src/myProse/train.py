import json
import os
import random
import re
import numpy as np
import math
import gc
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import time 
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
from esm.esm import pretrained
from model import SkipLSTM
from dataset import SCOPePairsDataset, LSTMDataset, SSADataset
from Bio import SeqIO
import logging
import datetime
from pandas import *
from utils import (
    L2_distance, 
    SSA_score_slow, 
    UPGMA, UPGMA_Kmeans,
    runcmd, 
    calculate_corr, 
    BisectingKmeans
)

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

def SSA_distance(x, batch_size=1024):
    dist_matrix = torch.zeros(len(x), len(x))
    dist_matrix.fill_diagonal_(10000)
    # print('Successfully Allocate Memory !!')
    dataset = SSADataset(x)
    dataLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        # num_workers=8,
    )
    with torch.no_grad():
        for seqA, seqB, X, Y in dataLoader:
            # seqA = seqA.cuda()
            # seqB = seqB.cuda()
            dist = SSA_score_slow(seqA, seqB)
            for score, x, y in zip(dist, X, Y):
                dist_matrix[x][y] = dist_matrix[y][x] = score
    return dist_matrix

def eval_prog(args):
    result = {}
    if args.eval_dataset == 'bb3_release':
        fasta_dir = Path("../../data/bb3_release")
    elif args.eval_dataset == 'homfam-small':
        fasta_dir = Path("../../data/homfam/small")
    elif args.eval_dataset == 'homfam-medium':
        fasta_dir = Path("../../data/homfam/medium")
    elif args.eval_dataset == 'homfam-large':
        fasta_dir = Path("../../data/homfam/large")
    elif args.eval_dataset == 'exthomfam-small':
        fasta_dir = Path("../../data/extHomFam-v3/small")
    elif args.eval_dataset == 'exthomfam-medium':
        fasta_dir = Path("../../data/extHomFam-v3/medium")
    elif args.eval_dataset == 'exthomfam-huge':
        fasta_dir = Path("../../data/extHomFam-v3/huge")
    elif args.eval_dataset == 'exthomfam-large':
        fasta_dir = Path("../../data/extHomFam-v3/large")
    elif args.eval_dataset == 'exthomfam-xlarge':
        fasta_dir = Path("../../data/extHomFam-v3/xlarge")
    elif args.eval_dataset == 'oxfam-small':
        fasta_dir = Path("../../data/oxfam/small")
    elif args.eval_dataset == 'oxfam-medium':
        fasta_dir = Path("../../data/oxfam/medium")
    elif args.eval_dataset == 'oxfam-large':
        fasta_dir = Path("../../data/oxfam/large")
    elif args.eval_dataset == 'ContTest-small':
        fasta_dir = Path("../../data/ContTest/data/small")
    elif args.eval_dataset == 'ContTest-medium':
        fasta_dir = Path("../../data/ContTest/data/medium")
    elif args.eval_dataset == 'ContTest-large':
        fasta_dir = Path("../../data/ContTest/data/large")
    else:
        raise NotImplementedError
    if 'ContTest' in args.eval_dataset:
        fasta_files = []
        for path in fasta_dir.iterdir():
            if path.is_dir() and "PF" in path.stem:
                prefix = path.stem.split('_')[0]
                fasta_files.append(path / f"{prefix}_unaligned.fasta")
    else:
        fasta_files = list(fasta_dir.glob('**/*.tfa'))
    for i, fastaFile in enumerate(tqdm(fasta_files, desc="Eval Guide Tree")):
        # print(f"Now processing file ({i + 1}/{len(list(fasta_dir.glob('**/*.tfa')))}) : {fastaFile.name}")
        logging.info(f"Now processing file ({i + 1}/{len(fasta_files)}) : {fastaFile.name}")
        raw_seqs = list(SeqIO.parse(fastaFile, 'fasta'))
        ## Alignment
        if args.align_prog == "clustalo":
            if args.eval_dataset == 'bb3_release':
                msf_path = args.msf_dir / f"{fastaFile.stem}.msf"
                runcmd(f"./clustalo --threads=8 --outfmt=msf --in {fastaFile.absolute().resolve()} --out {msf_path.absolute().resolve()} --force")
            else:
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_clustalo.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"./clustalo --threads=8 --in {fastaFile.absolute().resolve()} --out {pfa_path.absolute().resolve()} --force")
        elif args.align_prog == "mafft":
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_mafft.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
            ret = runcmd(f"./mafft --large --anysymbol --thread 8 {fastaFile.absolute().resolve()}").decode().split('\n')
            with open(pfa_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
        else:
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            if args.align_prog == "famsa":
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_famsa.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"famsa -keep-duplicates -gt upgma -t 8 {fastaFile.absolute().resolve()} {pfa_path.absolute().resolve()}")
            else:
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_tcoffee.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"t_coffee -reg -thread 8 -child_thread 8 -seq {fastaFile.absolute().resolve()} -nseq {min(200, len(raw_seqs) // 10)} -tree mbed -method mafftginsi_msa -outfile {pfa_path.absolute().resolve()}")
        ## Calculate Score
        if 'ContTest' in args.eval_dataset:
            continue
        if args.eval_dataset == 'bb3_release':
            xml_path = fastaFile.parents[0] / f"{fastaFile.stem}.xml"
            output = runcmd(f"bali_score {xml_path} {msf_path}").decode("utf-8").split('\n')[10]
            SP = float(output.split()[2])
            TC = float(output.split()[3])
        else:
            rfa_path = list(fasta_dir.glob(f"{fastaFile.stem}.rfa"))[0]
            rfa_pfa_path = args.msf_dir / f"{fastaFile.stem}_rfa.pfa"
            rfa_raw = list(SeqIO.parse(rfa_path, 'fasta'))
            pfa_raw = list(SeqIO.parse(pfa_path, 'fasta'))
            seq_in_ref = [str(ss.id) for ss in rfa_raw]
            with open(rfa_pfa_path, 'w') as f:
                for pfa in pfa_raw:
                    seq_name = str(pfa.id)
                    seq_data = str(pfa.seq)
                    if seq_name in seq_in_ref:
                        f.write(f">{seq_name}\n")
                        f.write(f"{seq_data}\n")
            raw_scores = runcmd(f"java -jar {args.fastSP_path.absolute().resolve()} -r {rfa_path.absolute().resolve()} -e {rfa_pfa_path.absolute().resolve()}").decode().split()
            SP = float(raw_scores[raw_scores.index('SP-Score') + 1])
            TC = float(raw_scores[raw_scores.index('TC') + 1])
        # print(f"SP-score = {SP}")
        # print(f"TC = {TC}")
        logging.info(f"SP-score = {SP}")
        logging.info(f"TC = {TC}")
        # Collect Score
        category = fastaFile.parents[0].name
        if category not in result:
            result[category] = {
                'SP' : [SP],
                'TC' : [TC]
            }
        else:
            result[category]['SP'].append(SP)
            result[category]['TC'].append(TC)
    
    final_result = {}
    if 'ContTest' in args.eval_dataset:
        # runcmd(f"cd {fasta_dir.absolute().resolve()}")
        # runcmd(f"runbenchmark -a {args.align_prog}")
        # csv = read_csv(f"results/results_{args.align_prog}_psicov.csv")
        # PFAM_ID = csv['PFAM_ID'].tolist()
        # PROTEIN_ID = csv[' PROTEIN_ID'].tolist()
        # SCORE = csv[' SCORE'].tolist()
        # total_score = 0
        # for pfam, prot, score in zip(PFAM_ID, PROTEIN_ID, SCORE):
        #     logging.info(f"{pfam}\t\t{prot}\t\t{score}")
        #     total_score += float(score)
        cat = fasta_dir.stem
        final_result[cat] = {
            "SCORE" : 0
        }
    else:
        for cat, value in result.items():
            final_result[cat] = {
                "SP" : sum(value['SP']) / len(value['SP']),
                "TC" : sum(value['TC']) / len(value['TC']),
            }
    return final_result

def eval_Kmeans(epoch, model, esm_alphabet, args):
    result = {}
    if args.eval_dataset == 'bb3_release':
        fasta_dir = Path("../../data/bb3_release")
    elif args.eval_dataset == 'homfam-small':
        fasta_dir = Path("../../data/homfam/small")
    elif args.eval_dataset == 'homfam-medium':
        fasta_dir = Path("../../data/homfam/medium")
    elif args.eval_dataset == 'homfam-large':
        fasta_dir = Path("../../data/homfam/large")
    elif args.eval_dataset == 'exthomfam-small':
        fasta_dir = Path("../../data/extHomFam-v3/small")
    elif args.eval_dataset == 'exthomfam-medium':
        fasta_dir = Path("../../data/extHomFam-v3/medium")
    elif args.eval_dataset == 'exthomfam-huge':
        fasta_dir = Path("../../data/extHomFam-v3/huge")
    elif args.eval_dataset == 'exthomfam-large':
        fasta_dir = Path("../../data/extHomFam-v3/large")
    elif args.eval_dataset == 'exthomfam-xlarge':
        fasta_dir = Path("../../data/extHomFam-v3/xlarge")
    elif args.eval_dataset == 'oxfam-small':
        fasta_dir = Path("../../data/oxfam/small")
    elif args.eval_dataset == 'oxfam-medium':
        fasta_dir = Path("../../data/oxfam/medium")
    elif args.eval_dataset == 'oxfam-large':
        fasta_dir = Path("../../data/oxfam/large")
    elif args.eval_dataset == 'ContTest-small':
        fasta_dir = Path("../../data/ContTest/data/small")
    elif args.eval_dataset == 'ContTest-medium':
        fasta_dir = Path("../../data/ContTest/data/medium")
    elif args.eval_dataset == 'ContTest-large':
        fasta_dir = Path("../../data/ContTest/data/large")
    else:
        raise NotImplementedError
    model = model.cuda()
    if 'ContTest' in args.eval_dataset:
        fasta_files = []
        for path in fasta_dir.iterdir():
            if path.is_dir() and "PF" in path.stem:
                prefix = path.stem.split('_')[0]
                fasta_files.append(path / f"{prefix}_unaligned.fasta")
    else:
        fasta_files = list(fasta_dir.glob('**/*.tfa'))
    for i, fastaFile in enumerate(tqdm(fasta_files, desc="Eval Guide Tree")):
        # print(f"Now processing file ({i + 1}/{len(list(fasta_dir.glob('**/*.tfa')))}) : {fastaFile.name}")
        logging.info(f"Now processing file ({i + 1}/{len(fasta_files)}) : {fastaFile.name}")
        ## Read sequences
        raw_seqs = list(SeqIO.parse(fastaFile, 'fasta'))
        seqs, avg_len, num_seqs = [], 0, len(raw_seqs)
        for idx, seq in enumerate(raw_seqs):
            seqs.append({
                'num' : idx + 1,
                'name' : str(idx + 1) if args.align_prog == 'mafft' else str(seq.id),
                'seq' : str(seq.seq),
                'embedding' : None,
            })
            avg_len += len(str(seq.seq))
        avg_len /= len(raw_seqs)
        # print(f"Average Sequence Length : {avg_len}")
        logging.info(f"Average Sequence Length : {avg_len}")
        ##
        sorted_seqs = sorted(seqs, key=lambda seq : len(seq['seq']), reverse=True)
        #### Release Memory ####
        del(raw_seqs)
        del(seqs)
        gc.collect()
        ########################
        queue, id2cluster = [], {}
        for i, seq in enumerate(sorted_seqs):
            if i > 0 and sorted_seqs[i]['seq'] == sorted_seqs[i-1]['seq']:
                queue[-1].append(sorted_seqs[i])
            else:
                queue.append([sorted_seqs[i]])
        unique_sorted_seqs = []
        for uniq in queue:
            unique_sorted_seqs.append(uniq[0])
            if len(uniq) > 1:
                id2cluster[uniq[0]['name']] = uniq
            else:
                id2cluster[uniq[0]['name']] = None
        unique_sorted_seqs.sort(key=lambda seq : seq['num'])
        # print(f"Unique sequences : {len(unique_sorted_seqs)} / {len(seqs)}")
        logging.info(f"Unique sequences : {len(unique_sorted_seqs)} / {num_seqs}")
        
        ## Create Dataset / Dataloader
        if esm_alphabet is not None:
            dataset = LSTMDataset([seq['seq'] for seq in unique_sorted_seqs], esm_alphabet)
        else:
            dataset = LSTMDataset([seq['seq'] for seq in unique_sorted_seqs])
        eval_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            # batch_sampler=dataset.batch_sampler(args.toks_per_batch_eval),
            shuffle=False,
            num_workers=8,
        )
        ## Create Embeddings
        embeddings, index = [], []
        start_time = time.time()
        with torch.no_grad():
            for i, (tokens, lengths, indices) in enumerate(tqdm(eval_loader, desc="Embed Sequences:")):
                tokens = tokens.cuda()
                emb = model(tokens, lengths)
                embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        for idx, emb in enumerate(embeddings):
            unique_sorted_seqs[idx]['embedding'] = emb
        # print(f"Finish embedding in {time.time() - start_time} secs.")
        logging.info(f"Finish embedding in {time.time() - start_time} secs.")
        
        centers, clusters = BisectingKmeans(unique_sorted_seqs)
        # print(f"Cluster Sizes : {[len(cl) for cl in clusters]}")
        logging.info(f"Cluster Sizes : {[len(cl) for cl in clusters]}")
        if len(centers) > 1:
            center_embeddings = torch.stack([cen['embedding'] for cen in centers], dim=0)
            ## Create Distance Matrix
            dist_matrix = torch.cdist(center_embeddings, center_embeddings, 2).fill_diagonal_(1000000000).cpu().numpy() / 50
            # dist_matrix = L2_distance(center_embeddings) / 50
        else:
            center_embeddings = None
            dist_matrix = None
        ## UPGMA / Output Guide Tree
        tree_path = args.tree_dir / f"{fastaFile.stem}.dnd"
        UPGMA_Kmeans(dist_matrix, clusters, id2cluster, tree_path, args.fasta_dir, args.dist_type)
        ####### Release Memory ######
        del embeddings              #
        del center_embeddings       #
        del unique_sorted_seqs      #
        del queue                   #
        del clusters                #
        del centers                 #
        del id2cluster              #
        gc.collect()                #
        #############################
        ## Alignment
        if args.align_prog == "clustalo":
            if args.eval_dataset == 'bb3_release':
                msf_path = args.msf_dir / f"{fastaFile.stem}.msf"
                runcmd(f"./clustalo --threads=8 --outfmt=msf --in {fastaFile.absolute().resolve()} --out {msf_path.absolute().resolve()} --guidetree-in {tree_path.absolute().resolve()} --force")
            else:
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_clustaloNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"./clustalo --threads=8 --in {fastaFile.absolute().resolve()} --out {pfa_path.absolute().resolve()} --guidetree-in {tree_path.absolute().resolve()} --force")
        elif args.align_prog == "mafft":
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            mafft_path = args.tree_dir / f"{fastaFile.stem}_mafft.dnd"
            pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_mafftNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
            ret = runcmd(f"./newick2mafft.rb {tree_path.absolute().resolve()}").decode().split('\n')
            with open(mafft_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
            ret = runcmd(f"./mafft --anysymbol --thread 8 --treein {mafft_path.absolute().resolve()} {fastaFile.absolute().resolve()}").decode().split('\n')
            with open(pfa_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
        else:
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            
            if args.align_prog == "famsa":
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_famsaNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"famsa -keep-duplicates -t 8 -gt import {tree_path.absolute().resolve()} {fastaFile.absolute().resolve()} {pfa_path.absolute().resolve()}")
            else:
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_tcoffeeNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"t_coffee -reg -thread 8 -child_thread 8 -seq {fastaFile.absolute().resolve()} -nseq {min(200, num_seqs // 10)} -tree {tree_path.absolute().resolve()} -method mafftgins1_msa -outfile {pfa_path.absolute().resolve()}")

        ## Calculate Score
        if 'ContTest' in args.eval_dataset:
            continue
        if args.eval_dataset == 'bb3_release':
            xml_path = fastaFile.parents[0] / f"{fastaFile.stem}.xml"
            output = runcmd(f"bali_score {xml_path} {msf_path}").decode("utf-8").split('\n')[10]
            SP = float(output.split()[2])
            TC = float(output.split()[3])
        else:
            rfa_path = list(fasta_dir.glob(f"{fastaFile.stem}.rfa"))[0]
            rfa_pfa_path = args.msf_dir / f"{fastaFile.stem}_rfa.pfa"
            rfa_raw = list(SeqIO.parse(rfa_path, 'fasta'))
            pfa_raw = list(SeqIO.parse(pfa_path, 'fasta'))
            seq_in_ref = [str(ss.id) for ss in rfa_raw]
            with open(rfa_pfa_path, 'w') as f:
                for pfa in pfa_raw:
                    seq_name = str(pfa.id)
                    seq_data = str(pfa.seq)
                    if seq_name in seq_in_ref:
                        f.write(f">{seq_name}\n")
                        f.write(f"{seq_data}\n")
            raw_scores = runcmd(f"java -jar {args.fastSP_path.absolute().resolve()} -r {rfa_path.absolute().resolve()} -e {rfa_pfa_path.absolute().resolve()}").decode().split()
            SP = float(raw_scores[raw_scores.index('SP-Score') + 1])
            TC = float(raw_scores[raw_scores.index('TC') + 1])
        # print(f"SP-score = {SP}")
        # print(f"TC = {TC}")
        logging.info(f"SP-score = {SP}")
        logging.info(f"TC = {TC}")
        # Collect Score
        category = fastaFile.parents[0].name
        if category not in result:
            result[category] = {
                'SP' : [SP],
                'TC' : [TC]
            }
        else:
            result[category]['SP'].append(SP)
            result[category]['TC'].append(TC)
    
    final_result = {}
    if 'ContTest' in args.eval_dataset:
        # runcmd(f"cd {fasta_dir.absolute().resolve()}")
        # runcmd(f"runbenchmark -a {args.align_prog}")
        # csv = read_csv(f"results/results_{args.align_prog}_psicov.csv")
        # PFAM_ID = csv['PFAM_ID'].tolist()
        # PROTEIN_ID = csv[' PROTEIN_ID'].tolist()
        # SCORE = csv[' SCORE'].tolist()
        # total_score = 0
        # for pfam, prot, score in zip(PFAM_ID, PROTEIN_ID, SCORE):
        #     logging.info(f"{pfam}\t\t{prot}\t\t{score}")
        #     total_score += float(score)
        cat = fasta_dir.stem
        final_result[cat] = {
            "SCORE" : 0
        }
    else:
        for cat, value in result.items():
            final_result[cat] = {
                "SP" : sum(value['SP']) / len(value['SP']),
                "TC" : sum(value['TC']) / len(value['TC']),
            }
    return final_result

def eval_guideTree_bb3_release(epoch, model, esm_alphabet, args):
    result = {}
    for i, fastaFile in enumerate(tqdm(list(args.eval_tree_dir.glob('**/*.tfa')), desc="Eval Guide Tree")):
        # print(f"Now processing file ({i + 1}/{len(list(args.inputFolder.glob('**/*.tfa')))}) : {fastaFile.name}")
        ## Read sequences
        raw_seqs = list(SeqIO.parse(fastaFile, 'fasta'))
        seqs = [str(seq.seq) for seq in raw_seqs]
        seqID = [str(seq.id) for seq in raw_seqs]
        ## Create Dataset / Dataloader
        if esm_alphabet is not None:
            dataset = LSTMDataset(seqs, esm_alphabet)
        else:
            dataset = LSTMDataset(seqs)
        eval_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size * 4,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            shuffle=False,
            # num_workers=8,
        )
        ## Create Embeddings
        embeddings = []
        with torch.no_grad():
            for i, (seqs, length) in enumerate(eval_loader):
                seqs = seqs.cuda()
                emb = model(seqs, length)
                for embedding in emb:
                    embeddings.append(embedding.cpu())
        # embeddings = torch.cat(embeddings, dim=0)
        ## Create Distance Matrix
        dist_matrix = SSA_distance(embeddings)
        ## UPGMA / Output Guide Tree
        tree_path = args.tree_dir / f"{fastaFile.stem}.dnd"
        UPGMA(dist_matrix, seqID, tree_path)
        ## Alignment
        msf_path = args.msf_dir / f"{fastaFile.stem}.msf"
        runcmd(f"clustalo --outfmt=msf --in {fastaFile.absolute().resolve()} --out {msf_path.absolute().resolve()} --guidetree-in {tree_path.absolute().resolve()} --force")
        ## Collect Score
        xml_path = fastaFile.parents[0] / f"{fastaFile.stem}.xml"
        output = runcmd(f"bali_score {xml_path} {msf_path}").decode("utf-8").split('\n')[10]
        SP = float(output.split()[2])
        TC = float(output.split()[3])
        category = fastaFile.parents[0].name
        if category not in result:
            result[category] = {
                'SP' : [SP],
                'TC' : [TC]
            }
        else:
            result[category]['SP'].append(SP)
            result[category]['TC'].append(TC)
    final_result = {}
    for cat, value in result.items():
        final_result[cat] = {
            "SP" : sum(value['SP']) / len(value['SP']),
            "TC" : sum(value['TC']) / len(value['TC']),
        }
    return final_result
    
def eval_corr(epoch, model, criterion, eval_loader, args):
    eval_loss, num_eval_data = 0, 0
    predicted, ground = [], []
    model.eval()
    with torch.no_grad():
        for seqs, length, score in tqdm(eval_loader, desc='Eval Epoch #{}'.format(epoch)):
            seqs = seqs.cuda()
            score = score.cuda()
            
            emb = model(seqs, length)
            logits = model.score(emb)
            
            for i in range(len(logits)):
                predicted.append(logits[i].detach().cpu())
                ground.append(score[i].detach().cpu())
            loss = criterion(logits, score)
            eval_loss += loss.item()
            num_eval_data += len(score)
    
    eval_loss = eval_loss / num_eval_data
    eval_spearman, eval_pearson = calculate_corr(ground, predicted)     
    np.save(args.plot_dir / f'ground-{epoch}.npy', np.array(ground))
    np.save(args.plot_dir / f'predicted-{epoch}.npy', np.array(predicted))
    return eval_loss, eval_spearman, eval_pearson
        
def train(epoch, model, criterion, optimizer, scheduler, train_loader, args):
    model.train()
    train_loss, num_train_data = 0, 0
    predicted, ground = [], []
    with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch)) as t:
        for batch_idx, (seqs, length, score) in enumerate(train_loader):
            seqs = seqs.cuda()
            score = score.cuda()
                
            emb = model(seqs, length)
            logits = model.score(emb)
                
            loss = criterion(logits, score)
            train_loss += loss.item()
            num_train_data += len(score)

            loss = loss / args.accum_steps
            loss.backward()
            
            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            for i in range(len(logits)):
                predicted.append(logits[i].detach().cpu())
                ground.append(score[i].detach().cpu())

            t.set_postfix({
                'loss' : train_loss / num_train_data,
                'lr' : optimizer.param_groups[0]['lr']
            })
            t.update(1)
            
    train_loss = train_loss / num_train_data
    train_spearman, train_pearson = calculate_corr(ground, predicted)
    np.save(args.plot_dir / f'train-ground-{epoch}.npy', np.array(ground))
    np.save(args.plot_dir / f'train-predicted-{epoch}.npy', np.array(predicted))
    return train_loss, train_spearman, train_pearson
            
def main(args):
    same_seed(args.seed)
    # torch.cuda.set_device(args.gpu)

    if args.embed_type == 'esm-43M':
        esm_model, esm_alphabet = pretrained.esm1_t6_43M_UR50S(args.esm_path)
        repr_layer, hidden_dim = 6, 768
    elif args.embed_type == 'esm-35M':
        esm_model, esm_alphabet = pretrained.esm2_t12_35M_UR50D(args.esm_path)
        repr_layer, hidden_dim = 12, 480
    elif args.embed_type == 'esm-150M':
        esm_model, esm_alphabet = pretrained.esm2_t30_150M_UR50D(args.esm_path)
        repr_layer, hidden_dim = 30, 640
    elif args.embed_type == 'esm-650M':
        esm_model, esm_alphabet = pretrained.esm2_t33_650M_UR50D(args.esm_path)
        repr_layer, hidden_dim = 33, 1280
    else:
        esm_model, esm_alphabet = None, None
        repr_layer, hidden_dim = 3, 1024

    # train_set = SCOPePairsDataset(
    #     args.train_dir,
    #     'train',
    #     args.train_distance_metric,
    #     esm_alphabet,
    # )
    # eval_set = SCOPePairsDataset(
    #     args.eval_dir,
    #     'eval',
    #     args.eval_distance_metric,
    #     esm_alphabet,
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     # batch_sampler=train_set.batch_sampler(args.toks_per_batch_train),
    #     batch_size=args.batch_size,
    #     collate_fn=train_set.collate_fn,
    #     pin_memory=True,
    #     shuffle=True,
    #     num_workers=8,
    # )
    # eval_loader = torch.utils.data.DataLoader(
    #     eval_set,
    #     # batch_sampler=eval_set.batch_sampler(args.toks_per_batch_eval),
    #     batch_size=args.batch_size * 6,
    #     collate_fn=eval_set.collate_fn,
    #     pin_memory=True,
    #     shuffle=False,
    #     num_workers=8,
    # )
    
    model = SkipLSTM.load_pretrained(
        args.lstm_path, 
        args.score_type, 
        esm_model, 
        repr_layer,
        hidden_dim,
        args.output_dim
    )
    # mse_criterion = torch.nn.MSELoss()
    # if args.no_decay_keys:
    #     keys = args.no_decay_keys.split('#')
    #     model_params = [
    #         {"params" : model.get_parameters(keys, mode='exclude'), "weight_decay" : args.weight_decay},
    #         {"params" : model.get_parameters(keys, mode='include'), "weight_decay" : 0},
    #     ]
    # else:
    #     model_params = model.parameters()
    # optimizer = torch.optim.Adam(model_params, lr=1)
    
    # T_max = len(train_loader) * args.num_epoch // args.accum_steps
    # warm_up_iter = int(T_max * args.warmup_ratio)
    # lr_min, lr_max = 1e-6, 2e-5
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, 
    #     lr_lambda = lambda cur_iter: lr_min + cur_iter / warm_up_iter * (lr_max - lr_min) if  cur_iter < warm_up_iter else \
    #     (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))
    #     # lr_max - (lr_max - lr_min) / (T_max - warm_up_iter) * (cur_iter - warm_up_iter)
    # )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=0.7)
    
    # result = eval_guideTree_bb3_release(
    #     0,
    #     model,
    #     esm_alphabet,
    #     args
    # )
    tot_start_time = time.time()
    if args.no_tree:
        result = eval_prog(args)
    else: 
        result = eval_Kmeans(
            0,
            model,
            esm_alphabet,
            args
        )
    # eval_loss, eval_spearman, eval_pearson = eval_corr(
    #     0, 
    #     model, 
    #     mse_criterion, 
    #     eval_loader, 
    #     args
    # )
    logging.info(f"========== Before Training ==========")
    # print(f"Evaluation Loss : {eval_loss}")
    # print(f"Evaluation Spearman Correlation : {eval_spearman}")
    # print(f"Evaluation Pearson Correlation : {eval_pearson}")
    logging.info(f"Guide Tree Evaluation : ")
    if "ContTest" in args.eval_dataset:
        logging.info(f"Category\t\tSCORE")
        for key, value in result.items():
            logging.info(f"{key}\t\t{value['SCORE']}")
    else:
        logging.info(f"Category\t\tSP\t\tTC")
        for key, value in result.items():
            logging.info(f"{key}\t\t{value['SP']}\t\t{value['TC']}")
    logging.info(f"Total Execution Time : {time.time() - tot_start_time} (s)")
    logging.info(f"===================================")
    
    # for epoch in range(args.num_epoch):
    #     train_loss, train_spearman, train_pearson = train(
    #         epoch + 1, 
    #         model, 
    #         mse_criterion, 
    #         optimizer, 
    #         scheduler, 
    #         train_loader, 
    #         args
    #     )
    #     result = eval_guideTree_bb3_release(
    #         epoch + 1,
    #         model,
    #         esm_alphabet,
    #         args
    #     )
    #     eval_loss, eval_spearman, eval_pearson = eval_corr(
    #         epoch + 1, 
    #         model, 
    #         mse_criterion, 
    #         eval_loader, 
    #         args
    #     )
        
    #     # TODO: Evaluation loop - calculate accuracy and save model weights
    #     print(f"========== Epoch {epoch + 1} ==========")
    #     print(f"Training Loss : {train_loss}")
    #     print(f"Training Spearman Correlation : {train_spearman}")
    #     print(f"Training Pearson Correlation : {train_pearson}")
    #     print(f"Evaluation Loss : {eval_loss}")
    #     print(f"Evaluation Spearman Correlation : {eval_spearman}")
    #     print(f"Evaluation Pearson Correlation : {eval_pearson}")
    #     print(f"Guide Tree Evaluation : ")
    #     print(f"Category\t\tSP\t\tTC")
    #     for key, value in result.items():
    #         print(f"{key}\t\t{value['SP']}\t\t{value['TC']}")
    #     print(f"===================================")
        
    #     torch.save({
    #         'epoch': epoch + 1,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'lr': optimizer.param_groups[0]['lr'],
    #         'train_loss': train_loss,
    #         'train_pearson': train_pearson,
    #         'train_spearman': train_spearman,
    #         'eval_loss': eval_loss,
    #         'eval_pearson': eval_pearson,
    #         'eval_spearman': eval_spearman,
    #     }, args.ckpt_dir / f'Epoch-{epoch + 1}.pt')
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/blast",
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/homfam/small",
    )
    parser.add_argument(
        "--lstm_path",
        type=Path,
        help="Path to pretrained LSTM model.",
        default="../../ckpt/prose/saved_models/prose_mt_3x1024.sav",
    )
    parser.add_argument(
        "--esm_path",
        type=str,
        help="Path to pretrained esm model.",
        default="./esm/ckpt/esm2_t33_650M_UR50D.pt",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./plot",
    )
    parser.add_argument(
        "--tree_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./trees",
    )
    parser.add_argument(
        "--msf_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./msf",
    )
    parser.add_argument(
        "--fasta_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./fasta",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Path to logs",
        default='./logs'
    )
    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.3106358771725278)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--no_decay_keys", type=str, default="RCNN#flatten#mask")
    parser.add_argument("--output_dim", type=int, default=100)

    # training
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--embed_type', type=str, default='LSTM', choices=['LSTM', 'esm-43M', 'esm-35M', 'esm-150M', 'esm-650M'])
    parser.add_argument('--score_type', type=str, default='MLP', choices=['SSA', 'L1', 'MLP'])
    parser.add_argument('--train_distance_metric', type=str, default='distance', choices=['distance', 'score'])
    parser.add_argument('--eval_distance_metric', type=str, default='distance', choices=['distance', 'score'])
    
    # eval
    # parser.add_argument("--eval_tree_dir", type=Path, default="../../data/bb3_release")
    parser.add_argument("--toks_per_batch_eval", type=int, default=16384)
    parser.add_argument("--newick2mafft_path", type=Path, default="./newick2mafft.rb")
    parser.add_argument("--fastSP_path", type=Path, default="./FastSP/FastSP.jar")
    parser.add_argument("--align_prog", type=str, default='clustalo', choices=["clustalo", "mafft", "famsa", "tcoffee"])
    ## WARNING : Do not use LCS if sequences are not unique !!
    parser.add_argument("--dist_type", type=str, default="NW", choices=["NW", "SW", "LCS"])
    parser.add_argument("--eval_dataset", type=str, default="bb3_release", choices=[
        "bb3_release", 
        "homfam-small", "homfam-medium", "homfam-large", 
        "oxfam-small", "oxfam-medium", "oxfam-large", 
        "exthomfam-small", "exthomfam-medium", "exthomfam-large", "exthomfam-huge", "exthomfam-xlarge",
        "ContTest-small", "ContTest-medium", "ContTest-large", 
    ])
    parser.add_argument("--no_tree", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # sys.stdout = open('./logging.txt', "w")
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.tree_dir.mkdir(parents=True, exist_ok=True)
    args.msf_dir.mkdir(parents=True, exist_ok=True)
    args.fasta_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.no_tree:
        log_filename = args.log_dir / datetime.datetime.now().strftime(f"{args.align_prog}_{args.eval_dataset}_%Y-%m-%d_%H_%M_%S.log")
    else:
        log_filename = args.log_dir / datetime.datetime.now().strftime(f"mix_{args.align_prog}_{args.eval_dataset}_%Y-%m-%d_%H_%M_%S.log")        
    logging.basicConfig(
        level=logging.INFO, 
        filename=log_filename, 
        filemode='w',
	    format='[%(asctime)s %(levelname)-8s] %(message)s',
	    datefmt='%Y%m%d %H:%M:%S',
	)
    main(args)