from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List
import subprocess
from subprocess import PIPE
from tqdm import tqdm

def runcmd(command):
    bash_command = command.split()
    ret = subprocess.run(bash_command, stdout=PIPE, stderr=PIPE)
    if ret.returncode == 0:
        return ret.stdout
    else:
        print("Error !!")
        return ret.stderr

def main(args):
    #runcmd(f"java -jar FastSP/FastSP.jar -r ../data/homfam/small/az.rfa -e ../msf/homfam/small/prose_mt_100/az.pfa")
    #runcmd(f"source /etc/profile")
    SP = 0
    for i, fastaFile in enumerate(tqdm(list(args.pred_dir.glob('**/*.pfa')))):
        name = fastaFile.stem
        ref = list(args.ref_dir.glob(f"{name}.rfa"))[0]
        seq_in_ref = []
        with open(ref, 'r') as f:
            line = f.readline()
            while line:
                if line[0] == '>':
                    seq_in_ref.append(line.strip('\n'))
                while True:
                    data = f.readline()
                    if not data or data[0] == '>':
                        break
                line = data
        with open(fastaFile, 'r') as f1, open(args.tmp_file_path / f"{name}.pfa", 'w') as f2:
            line = f1.readline()
            while line:
                seq_data, data = "", ""
                while True:
                    data = f1.readline()
                    if not data or data[0] == '>':
                        break
                    seq_data += data.strip("\n")
                if line.strip('\n') in seq_in_ref:
                    f2.write(line)
                    f2.write(seq_data + '\n')
                line = data
        
        raw_scores = runcmd(f"java -jar {args.prog_path.absolute().resolve()} -r {ref.absolute().resolve()} -e {(args.tmp_file_path / f'{name}.pfa').absolute().resolve()}").decode().split()
        print(float(raw_scores[raw_scores.index('SP-Score') + 1]))
        SP += float(raw_scores[raw_scores.index('SP-Score') + 1])
        runcmd(f"rm {(args.tmp_file_path / f'{name}.pfa').absolute().resolve()}")
    
    print(f"AVG SP score = {SP / len(list(args.pred_dir.glob('**/*.pfa')))}")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=Path,
        help="Path to predicted alignment",
        default="../msf/homfam/small/prose_mt_100",
    )
    parser.add_argument(
        "--ref_dir",
        type=Path,
        help="Path to reference alignment",
        default="../data/homfam/small",
    )
    parser.add_argument(
        "--prog_path",
        type=Path,
        help="Path to FastSP program",
        default="./FastSP/FastSP.jar",
    )
    parser.add_argument(
        "--tmp_file_path",
        type=Path,
        help="Path to temporary files",
        default="./",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)