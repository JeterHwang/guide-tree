import json
import os
import random
import numpy as np

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

def main(args):
    same_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.set_device(args.gpu)

    model = SkipLSTM.load_pretrained(args.model_path).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[20, 40], 
        gamma=args.gamma
    )
    criterion = torch.nn.MSELoss()

    train_set = SCOPePairsDataset(
        args.data_dir / 'train.json',
        'train'
    )
    eval_set = SCOPePairsDataset(
        args.data_dir / 'eval.json',
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
    for epoch in range(args.num_epoch):
        # TODO: Evaluation loop - calculate accuracy and save model weights
        eval_loss = 0
        predicted, ground = [], []
        model.eval()
        with torch.no_grad():
            for seqA, lenA, seqB, lenB, score in eval_loader:
                seqA = seqA.cuda()
                seqB = seqB.cuda()
                score = score.cuda()
                predict_similarity = model(seqA, lenA, seqB, lenB)
                for i in range(len(predict_similarity)):
                    predicted.append(predict_similarity[i].detach().cpu())
                    ground.append(score[i].detach().cpu())
                loss = criterion(predict_similarity, score)
                eval_loss += loss.item()

        eval_pearson, _ = pearsonr(np.array(ground), np.array(predicted))
        eval_spearman, _ = spearmanr(np.array(ground), np.array(predicted))
        print(f"Evaluation Loss : {eval_loss / len(eval_loader)}")
        print(f"Spearman Correlation : {eval_spearman}")
        print(f"Pearson Correlation : {eval_pearson}")

        model.train()
        train_loss, num_train_data = 0, 0
        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1)) as t:
            for seqA, lenA, seqB, lenB, score in train_loader:
                seqA = seqA.cuda()
                seqB = seqB.cuda()
                score = score.cuda()
                predict_similarity = model(seqA, lenA, seqB, lenB)

                optimizer.zero_grad()
                loss = criterion(predict_similarity, score)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                train_loss += loss.item()
                num_train_data += 1
                t.set_postfix({
                    'loss' : train_loss / num_train_data,
                    'lr' : optimizer.param_groups[0]['lr']
                })
                t.update(1)
        scheduler.step()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
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

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.3106358771725278)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)