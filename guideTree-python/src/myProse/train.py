import json
import os
import random
import numpy as np

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

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
    torch.cuda.set_device(args.gpu)

    model = SkipLSTM.load_pretrained(args.ckpt_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = torch.nn.MSELoss()

    train_set = SCOPePairsDataset(
        args.data_dir / './astral-scopedom-seqres-gd-all-2.08-stable.fa',
        args.data_dir / 'train.json',
        'train'
    )
    train_loader = torch.utils.data.Dataloader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collade_fn,
        shuffle=True,
        pin_memory=True
    )
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        for seqA, lenA, seqB, lenB, score in tqdm(train_loader):
            pass
        # TODO: Evaluation loop - calculate accuracy and save model weights
        


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)