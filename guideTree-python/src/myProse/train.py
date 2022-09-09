import json
import os
import random
import re
import numpy as np
import math
import itertools
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
    else:
        raise NotImplementedError
    for i, fastaFile in enumerate(tqdm(list(fasta_dir.glob('**/*.tfa')), desc="Eval Guide Tree")):
        print(f"Now processing file ({i + 1}/{len(list(fasta_dir.glob('**/*.tfa')))}) : {fastaFile.name}")
        ## Alignment
        if args.align_prog == "clustalo":
            if args.eval_dataset == 'bb3_release':
                msf_path = args.msf_dir / f"{fastaFile.stem}.msf"
                runcmd(f"clustalo --outfmt=msf --in {fastaFile.absolute().resolve()} --out {msf_path.absolute().resolve()} --force")
            else:
                pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"clustalo --in {fastaFile.absolute().resolve()} --out {pfa_path.absolute().resolve()} --force")
        elif args.align_prog == "mafft":
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            ret = runcmd(f"mafft {fastaFile.absolute().resolve()}").decode().split('\n')
            with open(pfa_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
        else:
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            runcmd(f"famsa -gt upgma -t 8 {fastaFile.absolute().resolve()} {pfa_path.absolute().resolve()}")
        ## Calculate Score
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
        print(f"SP-score = {SP}")
        print(f"TC = {TC}")
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
    else:
        raise NotImplementedError

    for i, fastaFile in enumerate(tqdm(list(fasta_dir.glob('**/*.tfa')), desc="Eval Guide Tree")):
        print(f"Now processing file ({i + 1}/{len(list(fasta_dir.glob('**/*.tfa')))}) : {fastaFile.name}")
        ## Read sequences
        raw_seqs = list(SeqIO.parse(fastaFile, 'fasta'))
        print(f"Number of Sequences : {len(raw_seqs)}")
        seqs = [str(seq.seq) for seq in raw_seqs]
        seqID = [str(seq.id) for seq in raw_seqs]
        ## Create Dataset / Dataloader
        if esm_alphabet is not None:
            dataset = LSTMDataset(seqs, esm_alphabet)
        else:
            dataset = LSTMDataset(seqs)
        eval_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=20,
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
        print(f"Finish embedding in {time.time() - start_time} secs.")
        Nodes = []
        for idx, (seq, ID, emb) in enumerate(zip(seqs, seqID, embeddings)):
            Nodes.append({
                'num' : idx + 1,
                'name' : ID,
                'seq' : seq,
                'embedding' : emb,
            })
        centers, clusters = BisectingKmeans(Nodes)
        if len(centers) > 1:
            center_embeddings = torch.stack([cen['embedding'] for cen in centers], dim=0)
            ## Create Distance Matrix
            dist_matrix = L2_distance(center_embeddings) / 50
        else:
            dist_matrix = None
        ## UPGMA / Output Guide Tree
        tree_path = args.tree_dir / f"{fastaFile.stem}.dnd"
        UPGMA_Kmeans(dist_matrix, clusters, tree_path, args.fasta_dir, args.align_prog)
        ## Alignment
        if args.align_prog == "clustalo":
            if args.eval_dataset == 'bb3_release':
                msf_path = args.msf_dir / f"{fastaFile.stem}.msf"
                runcmd(f"clustalo --outfmt=msf --in {fastaFile.absolute().resolve()} --out {msf_path.absolute().resolve()} --guidetree-in {tree_path.absolute().resolve()} --force")
            else:
                pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"clustalo --in {fastaFile.absolute().resolve()} --out {pfa_path.absolute().resolve()} --guidetree-in {tree_path.absolute().resolve()} --force")
        elif args.align_prog == "mafft":
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            mafft_path = args.tree_dir / f"{fastaFile.stem}_mafft.dnd"
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            ret = runcmd(f"./newick2mafft.rb {tree_path.absolute().resolve()}").decode().split('\n')
            with open(mafft_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
            ret = runcmd(f"mafft --thread 8 --treein {mafft_path.absolute().resolve()} {fastaFile.absolute().resolve()}").decode().split('\n')
            with open(pfa_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
        else:
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            runcmd(f"famsa -t 8 -keep-duplicates -gt import {tree_path.absolute().resolve()} {fastaFile.absolute().resolve()} {pfa_path.absolute().resolve()}")

        ## Calculate Score
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
        print(f"SP-score = {SP}")
        print(f"TC = {TC}")
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
    ).cuda()
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
    print(f"========== Before Training ==========")
    # print(f"Evaluation Loss : {eval_loss}")
    # print(f"Evaluation Spearman Correlation : {eval_spearman}")
    # print(f"Evaluation Pearson Correlation : {eval_pearson}")
    print(f"Guide Tree Evaluation : ")
    print(f"Category\t\tSP\t\tTC")
    for key, value in result.items():
        print(f"{key}\t\t{value['SP']}\t\t{value['TC']}")
    print(f"===================================")
    
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

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.3106358771725278)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--no_decay_keys", type=str, default="RCNN#flatten#mask")
    parser.add_argument("--output_dim", type=int, default=1024)

    # training
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
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
    parser.add_argument("--align_prog", type=str, default='clustalo', choices=["clustalo", "mafft", "famsa"])
    parser.add_argument("--eval_dataset", type=str, default="bb3_release", choices=["bb3_release", "homfam-small", "homfam-medium", "homfam-large"])
    parser.add_argument("--no_tree", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    sys.stdout = open('./logging.txt', "w")
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.tree_dir.mkdir(parents=True, exist_ok=True)
    args.msf_dir.mkdir(parents=True, exist_ok=True)
    args.fasta_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)