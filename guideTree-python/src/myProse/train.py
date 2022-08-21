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
import torch.nn.functional as F
from tqdm import trange, tqdm
import sys
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

def train(model, class_criterion, mse_criterion, optimizer, scheduler, train_loader, eval_loader, args):
    for epoch in range(args.num_epoch):
        # TODO: Evaluation loop - calculate accuracy and save model weights
        eval_loss, correct, num_eval_data = 0, 0, 0
        predicted, ground, SSA = [], [], []
        model.eval()
        with torch.no_grad():
            for seqs, lens, score in tqdm(eval_loader, desc='Eval Epoch #{}'.format(epoch + 1)):
                seqs = seqs.cuda()
                score = score.cuda()
                # level = level.cuda()
                batch_size = seqs.size()[0] // 2
                emb = model(seqs, lens)
                logits2 = model.score(emb)
                # prob = F.softmax(logits1, dim=1)
                # levels = torch.arange(10).to(prob.device).float()
                # predict_similarity = torch.sum(prob * levels, 1)
                for i in range(len(logits2)):
                    predicted.append(logits2[i].detach().cpu())
                    ground.append(score[i].detach().cpu())
                loss = mse_criterion(logits2, score)
                eval_loss += loss.item()
                # _, predicted_label = torch.max(logits1, 1)
                # correct += torch.sum(predicted_label == level)
                num_eval_data += len(score)

        eval_pearson, _ = pearsonr(np.array(ground), np.array(predicted))
        eval_spearman, _ = spearmanr(np.array(ground), np.array(predicted))
        np.save(args.plot_dir / f'ground-{epoch}.npy', np.array(ground))
        np.save(args.plot_dir / f'predicted-{epoch}.npy', np.array(predicted))
        print(f"Evaluation Loss : {eval_loss / num_eval_data}")
        # print(f"Evaluationo Acc : {correct / num_train_data}")
        print(f"Spearman Correlation : {eval_spearman}")
        print(f"Pearson Correlation : {eval_pearson}")

        model.train()
        train_loss, num_train_data, correct = 0, 0, 0
        predicted_cls, predicted_mse, ground = [], [], []
        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1)) as t:
            for seqs, lens, score in train_loader:
                seqs = seqs.cuda()
                score = score.cuda()
                # level = level.cuda()
                emb = model(seqs, lens)
                logits2 = model.score(emb)
                
                optimizer.zero_grad()
                # classification_loss = class_criterion(logits1, level)
                mse_loss = mse_criterion(logits2, score)
                loss = mse_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0, norm_type=2)
                optimizer.step()

                # prob = F.softmax(logits1, dim=1)
                # levels = torch.arange(10).to(prob.device).float()
                # predict_similarity = torch.sum(prob * levels, 1)
                for i in range(len(logits2)):
                    # predicted_cls.append(predict_similarity[i].detach().cpu())
                    predicted_mse.append(logits2[i].detach().cpu())
                    ground.append(score[i].detach().cpu())

                train_loss += loss.item()
                # _, predicted_label = torch.max(logits1, 1)
                # correct += torch.sum(predicted_label == level)
                num_train_data += len(score)
                t.set_postfix({
                    # 'cls_acc' : correct.item() / num_train_data,
                    'loss' : train_loss / num_train_data,
                    'lr' : optimizer.param_groups[0]['lr']
                })
                t.update(1)
                scheduler.step()

            train_pearson, _ = pearsonr(np.array(ground), np.array(predicted_mse))
            train_spearman, _ = spearmanr(np.array(ground), np.array(predicted_mse))
            np.save(args.plot_dir / f'train-ground-{epoch}.npy', np.array(ground))
            np.save(args.plot_dir / f'train-predicted-{epoch}.npy', np.array(predicted_mse))
            print(f"Training Spearman Correlation : {train_spearman}")
            print(f"Training Pearson Correlation : {train_pearson}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_pearson': train_pearson,
            'train_spearman': train_spearman,
            'eval_loss': eval_loss,
            'eval_pearson': eval_pearson,
            'eval_spearman': eval_spearman,
        }, args.ckpt_dir / f'Epoch-{epoch}.pt')

def main(args):
    same_seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    
    train_set = SCOPePairsDataset(
        args.train_dir,
        'train',
        args.train_distance_metric,
    )
    eval_set = SCOPePairsDataset(
        args.eval_dir,
        'eval',
        args.eval_distance_metric,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        # batch_sampler=train_set.batch_sampler(args.toks_per_batch_train),
        batch_size=args.batch_size,
        collate_fn=train_set.collate_fn,
        pin_memory=True,
        shuffle=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        # batch_sampler=eval_set.batch_sampler(args.toks_per_batch_eval),
        batch_size=args.batch_size * 3,
        collate_fn=eval_set.collate_fn,
        pin_memory=True,
        shuffle=False,
    )
    
    model = SkipLSTM.load_pretrained(args.model_path, args.score_type).cuda()
    class_criterion = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=args.weight_decay)
    
    T_max = len(train_loader) * args.num_epoch
    warm_up_iter = int(T_max * args.warmup_ratio)
    lr_min, lr_max = 2e-6, 2e-5
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, 
    #     lr_lambda = lambda cur_iter: lr_min + cur_iter / warm_up_iter * (lr_max - lr_min) if  cur_iter < warm_up_iter else \
    #     (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))
    #     # lr_max - (lr_max - lr_min) / (T_max - warm_up_iter) * (cur_iter - warm_up_iter)
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=0.7)
    train(model, class_criterion, mse_criterion, optimizer, scheduler, train_loader, eval_loader, args)

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
        default="./data/blast",
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--toks_per_batch_train", type=int, default=16384)
    parser.add_argument("--toks_per_batch_eval", type=int, default=9216)

    # training
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_epoch", type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--score_type', type=str, default='SSA', choices=['SSA', 'L1', 'MLP'])
    parser.add_argument('--train_distance_metric', type=str, default='distance', choices=['distance', 'score'])
    parser.add_argument('--eval_distance_metric', type=str, default='distance', choices=['distance', 'score'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    sys.stdout = open('./logging.txt', "w")
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)