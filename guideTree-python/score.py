from pathlib import Path
from argparse import ArgumentParser, Namespace
import subprocess
from subprocess import PIPE
import csv
from tqdm import tqdm

def runcmd(command):
    ret = subprocess.run(command, stdout=PIPE, stderr=PIPE)
    if ret.returncode == 0:
        print("Success !!")
        return ret.stdout
    else:
        print("Error !!")
        return ret.stderr

def main(args):
    args.tree_dir = args.tree_dir / args.prog_type
    args.tree_dir.mkdir(parents=True, exist_ok=True)
    args.msf_dir = args.msf_dir / args.prog_type
    args.msf_dir.mkdir(parents=True, exist_ok=True)
    args.csv_path.mkdir(parents=True, exist_ok=True)

    # Stage 1 : Produce guided trees
    if args.prog_type in ['mafft', 'clustal', 'muscle']:
        for fastaFile in tqdm(list(args.data_dir.glob('**/*.tfa'))):
            postfix = fastaFile.stem
            if args.prog_type == 'mafft':
                oldTreeName = postfix + '.tfa.tree'
                newTreeName = postfix + '_mafft.tree'
                runcmd(f"mafft --retree 0 --treeout --localpair {str(fastaFile.absolute().resolve())}")
                runcmd(f"mv {str((args.data_dir / oldTreeName).absolute().resolve())} {str((args.tree_dir / newTreeName).absolute().resolve())}")
            elif args.prog_type == 'clustal':
                treeName = postfix + '_clustal.dnd'
                runcmd(f"clustalo --outfmt=msf --in {str(fastaFile.absolute().resolve())} --guidetree-out {str((args.tree_dir / treeName).absolute().resolve())} --force")
            elif args.prog_type == 'muscle':
                treeName = postfix + '_muscle.phy'
                runcmd(f"muscle3.8.31_i86linux -msf -in {str(fastaFile.absolute().resolve())} -tree2 {str((args.tree_dir / treeName).absolute().resolve())}")
            else:
                raise NotImplementedError
    else:
        if args.prog_type == 'mBed':
            runcmd(f"python main.py --embedding mBed --outputFolder {str(args.tree_dir.absolute().resolve())}")
        elif args.prog_type == 'esm-43M':
            runcmd(f"python main.py --embedding esm --toks_per_batch 4096 --esm_ckpt ./ckpt/esm/esm1_t6_43M_UR50S.pt --outputFolder {str(args.tree_dir.absolute().resolve())}")
        elif args.prog_type == 'esm-650M':
            runcmd(f"python main.py --embedding esm --toks_per_batch 4096 --esm_ckpt ./ckpt/esm/esm1b_t33_650M_UR50S.pt --outputFolder {str(args.tree_dir.absolute().resolve())}")
        else:
            raise NotImplementedError

    # stage 2 : generate msf files
    for fastaFile in tqdm(list(args.data_dir.glob('**/*.tfa'))):
        name = fastaFile.stem
        msf = name + '.msf'

        if args.prog_type == 'mBed':
            match = list(args.tree_dir.glob(f"*{name}_mBed.dnd"))[0]
        elif args.prog_type == 'esm-43M':
            match = list(args.tree_dir.glob(f"*{name}_esm.dnd"))[0]
        elif args.prog_type == 'esm-650M':
            match = list(args.tree_dir.glob(f"*{name}_esm.dnd"))[0]
        elif args.prog_type == 'clustal':
            match = list(args.tree_dir.glob(f"*{name}_clustal.dnd"))[0]
        elif args.prog_type == 'muscle':
            match = list(args.tree_dir.glob(f"*{name}_muscle.phy"))[0]
        elif args.prog_type == 'mafft':
            match = list(args.tree_dir.glob(f"*{name}.tree"))[0]
        else:
            print(f"Tree type {args.prog_type} is not supported !!")
            raise NotImplementedError
        runcmd(f"clustalo --outfmt=msf --in {str((args.data_dir / oldTreeName).absolute().resolve())} --out {str((args.msf_dir / msf).absolute().resolve())} --guidetree-in {str(match.absolute().resolve())} --force")

    # Stage 3 : Scoring msf files
    csvFileName = args.csv_path / f"scores_balibase_{args.prog_type}.csv"
    with open(csvFileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for xmlFile in tqdm(list(args.data_dir.glob('**/*.xml'))):
            name = xmlFile.stem
            msf = name + '.msf'

            result = runcmd(f"bali_score {str(xmlFile.absolute().resolve())} {str((args.msf_dir / msf).absolute().resolve())} | grep auto")
            print(result)
            writer.writerow([result])
            
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--prog_type",
        type=str,
        required=True,
        help="Path to scoring programs",
        choices=["mafft", 'mBed', 'esm-43M', 'esm-650M', 'clustal', 'muscle'],
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Path to scoring programs",
        default="./data/bb3_release",
    )
    parser.add_argument(
        "--tree_dir",
        type=Path,
        help="Path to tree files",
        default="./output/bb3_release/trees",
    )
    parser.add_argument(
        "--msf_dir",
        type=Path,
        help="Path to msf files",
        default="./output/bb3_release/msf",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        help="Path to csv score file",
        default="./output/bb3_release/scores",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)