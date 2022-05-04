from pathlib import Path
from argparse import ArgumentParser, Namespace
from secrets import choice
from typing import List

def main(args):
    tree_dir = args.tree_dir / args.prog_type
    tree_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.bash_path / f"{args.prog_type}_tree_gen.sh", 'w') as f:
        fasta_num = len(list(args.data_dir.glob('**/*.tfa')))

        for i, fastaFile in enumerate(list(args.data_dir.glob('**/*.tfa'))):
            f.write(f"echo -ne '{'#' * (round(i / fasta_num * 100))}{' ' * (100 - round(i / fasta_num * 100))}({round(i / fasta_num * 100)}%)\\r'\n")
            f.write("sleep 1\n")
            
            name = fastaFile.stem
            postfix = name + '.msf'

            if args.prog_type == 'mafft':
                f.write(f"./run_guideTree_aln.sh -p mafft -i {str(fastaFile.absolute().resolve())} -o {str((args.bash_path / postfix).absolute().resolve())} -b {str((tree_dir / f'{name}.tree').absolute().resolve())}\n")
                f.write(f"mv {str(fastaFile.absolute().resolve())}.tree {str((tree_dir / f'{name}.tree').absolute().resolve())}\n")
            elif args.prog_type == 'clustal':
                f.write(f"./run_guideTree_aln.sh -p clustal -i {str(fastaFile.absolute().resolve())} -o {str((args.bash_path / postfix).absolute().resolve())} -b {str((tree_dir / f'{name}.dnd').absolute().resolve())}\n")
            elif args.prog_type == 'muscle':
                f.write(f"./run_guideTree_aln.sh -p muscle -i {str(fastaFile.absolute().resolve())} -o {str((args.bash_path / postfix).absolute().resolve())} -b {str((tree_dir / f'{name}.phy').absolute().resolve())}\n")
            else:
                raise NotImplementedError
            f.write(f"rm {str((args.bash_path / postfix).absolute().resolve())}\n")
                    
        f.write(f"echo -ne '{'#' * 100}(100%)\\r'\n")
        f.write("echo -ne '\\n'\n")
            
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Path to fasta files",
        default="../data/bb3_release",
    )
    parser.add_argument(
        "--tree_dir",
        type=Path,
        help="Path to generated trees",
        default="../trees/bb3_release/",
    )
    parser.add_argument(
        "--bash_path",
        type=Path,
        help="Path to store generated bash file",
        default="./",
    )
    parser.add_argument(
        "--prog_type",
        type=str,
        help="Program type",
        default='mafft',
        #choice=['mafft', 'clustal', 'muscle'],
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)