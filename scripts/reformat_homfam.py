import subprocess
from subprocess import PIPE
from pathlib import Path
from tqdm import tqdm
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
    for i, content in enumerate(tqdm(list(args.data_dir.glob('**/*')), desc="Eval Guide Tree")):
        if content.is_file() and content.parent.stem != 'ref':
            tfa_dir = args.target_dir / content.parent.stem
            tfa_dir.mkdir(parents=True, exist_ok=True)
            tfa_path = tfa_dir / f"{content.stem}.tfa"
            runcmd(f"cp {content.absolute().resolve()} {tfa_path.absolute().resolve()}")
            ref_path = content.parent.parent / "ref" / content.stem
            rfa_path = tfa_dir / f"{content.stem}.rfa"
            runcmd(f"cp {ref_path.absolute().resolve()} {rfa_path.absolute().resolve()}")