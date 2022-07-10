import json
import os
import random
import numpy as np
import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from scipy.stats import pearsonr, spearmanr
from sklearn.utils import shuffle
import torch
from tqdm import trange, tqdm

from model import SkipLSTM
from dataset import SCOPePairsDataset

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

def train(model, criterion, optimizer, scheduler, train_loader, eval_loader, args):
    for epoch in range(args.num_epoch):
        # TODO: Evaluation loop - calculate accuracy and save model weights
        eval_loss = 0
        predicted, ground, SSA = [], [], []
        model.eval()
        with torch.no_grad():
            for seqs, lens, score in tqdm(eval_loader, desc='Eval Epoch #{}'.format(epoch + 1)):
                seqs = seqs.cuda()
                score = score.cuda()
                predict_similarity = model(seqs, lens)
                SSA_score = model.SSA_score(seqs, lens)
                for i in range(len(predict_similarity)):
                    predicted.append(predict_similarity[i].detach().cpu())
                    ground.append(score[i].detach().cpu())
                    SSA.append(SSA_score[i].detach().cpu())
                loss = criterion(predict_similarity, score)
                eval_loss += loss.item()

        eval_pearson, _ = pearsonr(np.array(ground), np.array(predicted))
        eval_spearman, _ = spearmanr(np.array(ground), np.array(predicted))
        SSA_pearson, _ = pearsonr(np.array(ground), np.array(SSA))
        SSA_spearman, _ = spearmanr(np.array(ground), np.array(SSA))
        np.save(args.plot_dir / f'ground-{epoch}.npy', np.array(ground))
        np.save(args.plot_dir / f'predicted-{epoch}.npy', np.array(predicted))
        print(f"Evaluation Loss : {eval_loss / len(eval_loader)}")
        print(f"Spearman Correlation : {eval_spearman}")
        print(f"Pearson Correlation : {eval_pearson}")
        print(f"SSA Spearman Correlation : {SSA_spearman}")
        print(f"SSA Pearson Correlation : {SSA_pearson}")

        model.train()
        train_loss, num_train_data = 0, 0
        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1)) as t:
            for seqs, lens, score in train_loader:
                seqs = seqs.cuda()
                score = score.cuda()
                predict_similarity = model(seqs, lens)

                optimizer.zero_grad()
                loss = criterion(predict_similarity, score)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0, norm_type=2)
                optimizer.step()

                train_loss += loss.item()
                num_train_data += len(score)
                t.set_postfix({
                    'loss' : train_loss / num_train_data,
                    'lr' : optimizer.param_groups[0]['lr']
                })
                t.update(1)
                scheduler.step()

def eval_SSA(model, criterion, eval_loader, args):
    eval_loss, predicted, ground = 0, [], []
    model.eval()
    with torch.no_grad():
        for seqs, lens, score in tqdm(eval_loader, desc='Eval SSA Score'):
            seqs = seqs.cuda()
            score = score.cuda()
            predict_similarity = model.SSA_score(seqs, lens)
            for i in range(len(predict_similarity)):
                predicted.append(predict_similarity[i].detach().cpu())
                ground.append(score[i].detach().cpu())
            loss = criterion(predict_similarity, score)
            eval_loss += loss.item()
    predicted = np.array(predicted)
    ground = np.array(ground)
    eval_pearson, _ = pearsonr(ground, 20 * predicted / np.max(predicted))
    eval_spearman, _ = spearmanr(ground, 20 * predicted / np.max(predicted))
    # np.save(args.plot_dir / f'ground-{epoch}.npy', np.array(ground))
    # np.save(args.plot_dir / f'predicted-{epoch}.npy', np.array(predicted))
    print(f"Evaluation Loss : {eval_loss / len(eval_loader)}")
    print(f"Spearman Correlation : {eval_spearman}")
    print(f"Pearson Correlation : {eval_pearson}")

def main(args):
    same_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.set_device(args.gpu)

    train_set = SCOPePairsDataset(
        args.train_dir / 'train.json',
        'train'
    )
    eval_set = SCOPePairsDataset(
        args.eval_dir / 'eval.json',
        'eval'
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collade_fn,
        shuffle=True,
        pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=args.batch_size,
        collate_fn=eval_set.collade_fn,
        pin_memory=True
    )
    
    model = SkipLSTM.load_pretrained(args.model_path).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    
    T_max = len(train_loader) * args.num_epoch
    warm_up_iter = int(T_max * args.warmup_ratio)
    lr_min, lr_max = 1e-5, 4e-4
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda = lambda cur_iter: lr_min + cur_iter / warm_up_iter * (lr_max - lr_min) if  cur_iter < warm_up_iter else \
        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))
    )
    
    #eval_SSA(model, criterion, eval_loader, args)
    train(model, criterion, optimizer, scheduler, train_loader, eval_loader, args)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to pretrained model.",
        default="../../ckpt/prose/saved_models/prose_dlm_3x1024.sav",
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

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.3106358771725278)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # training
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    main(args)