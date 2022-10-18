import subprocess
from subprocess import PIPE
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
from argparse import ArgumentParser, Namespace

def runcmd(command):
    bash_command = command.split()
    ret = subprocess.run(bash_command, stdout=PIPE, stderr=PIPE)
    if ret.returncode == 0:
        return ret.stdout
    else:
        print(f"Error : {ret.stderr}")
        return ret.stderr

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/blast",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/blast",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg()
    for i, dir in enumerate(tqdm(args.data_dir.iterdir(), desc="Eval Guide Tree")):
        if dir.is_dir() and "PF" in dir.stem:
            pfamID = dir.stem.split('_')[0]
            seqs = list(SeqIO.parse(dir / f"{pfamID}_unaligned.fasta", 'fasta'))
            if len(seqs) < 3000:
                cat = 'small'
            elif len(seqs) <= 10000:
                cat = 'medium'
            else:
                cat = 'large'
            target_dir = args.data_dir / cat 
            target_dir.mkdir(parents=True, exist_ok=True)
            for fastaFile in dir.glob("*.fasta"):
                if pfamID not in fastaFile.stem:
                    protID = fastaFile.stem
            for cmFile in dir.glob("*.cm"):
                unknown = cmFile.stem
            runcmd(f"cp -r {dir.absolute().resolve()} {target_dir.absolute().resolve()}/")
            with open(target_dir / 'datasets.txt', 'a') as f:
                f.write(f"{pfamID}  {protID}  {unknown}\n")


