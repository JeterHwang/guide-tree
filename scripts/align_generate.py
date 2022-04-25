from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List

def main(args):
    output_dirs = [args.output_dir / tree_type for tree_type in args.tree_type]
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.bash_path, 'w') as f:
        fasta_num = len(list(args.data_dir.glob('**/*.tfa')))

        for i, fastaFile in enumerate(list(args.data_dir.glob('**/*.tfa'))):
            f.write(f"echo -ne '{'#' * (round(i / fasta_num * 100))}{' ' * (100 - round(i / fasta_num * 100))}({round(i / fasta_num * 100)}%)\\r'\n")
            f.write("sleep 1\n")
            
            name = fastaFile.stem
            postfix = name + '.msf'

            tree_dirs = [args.tree_dir / tree_type for tree_type in args.tree_type]
            matches = []
            for tree_type, tree_dir in zip(args.tree_type, tree_dirs):
                if tree_type == 'mBed':
                    matches.append(list(tree_dir.glob(f"*{name}_mBed.dnd"))[0])
                elif tree_type == 'esm-43M':
                    matches.append(list(tree_dir.glob(f"*{name}_esm.dnd"))[0])
                elif tree_type == 'clustal':
                    matches.append(list(tree_dir.glob(f"*{name}_clustal.dnd"))[0])
                elif tree_type == 'muscle':
                    matches.append(list(tree_dir.glob(f"*{name}_muscle.phy"))[0])
                else:
                    print(f"Tree type {tree_type} is not supported !!")
                    raise NotImplementedError
            
            for output_dir, match in zip(output_dirs, matches):
                f.write(f"./run_guideTree_aln.sh -p clustal -i {str(fastaFile.absolute().resolve())} -o {str((output_dir / postfix).absolute().resolve())} -a {str(match.absolute().resolve())}\n")    
            
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
        help="Path to run_alignment.sh",
        default="run_alignment.sh",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to msf output directory",
        default="../msf/bb3_release",
    )
    parser.add_argument(
        "--tree_type",
        type=str,
        nargs='+',
        help="tree file sources to read",
        default=['mBed', 'clustal', 'muscle', 'esm-43M'],
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)