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
            if args.output_format == 'msf':
                postfix = name + '.msf'
            elif args.output_format == 'fasta':
                postfix = name + '.pfa'
            else:
                raise NotImplementedError

            tree_dirs = [args.tree_dir / tree_type for tree_type in args.tree_type]
            matches = []
            for tree_type, tree_dir in zip(args.tree_type, tree_dirs):
                if tree_type == 'mBed':
                    matches.append(list(tree_dir.glob(f"*{name}_mBed.dnd"))[0])
                elif tree_type == 'esm-43M':
                    matches.append(list(tree_dir.glob(f"*{name}_esm.dnd"))[0])
                elif tree_type == 'esm-650M':
                    matches.append(list(tree_dir.glob(f"*{name}_esm.dnd"))[0])
                elif tree_type == 'prose_mt_100':
                    matches.append(list(tree_dir.glob(f"*{name}_prose_mt.dnd"))[0])
                elif tree_type == 'prose_mt_6K':
                    matches.append(list(tree_dir.glob(f"*{name}_prose_mt.dnd"))[0])
                elif tree_type == 'clustal':
                    matches.append(list(tree_dir.glob(f"*{name}_clustal.dnd"))[0])
                elif tree_type == 'muscle':
                    matches.append(list(tree_dir.glob(f"*{name}_muscle.phy"))[0])
                elif tree_type == 'mafft':
                    matches.append(list(tree_dir.glob(f"*{name}.tree"))[0])
                else:
                    print(f"Tree type {tree_type} is not supported !!")
                    raise NotImplementedError
            
            for output_dir, match in zip(output_dirs, matches):
                f.write(f"./run_guideTree_aln.sh -p clustal -f {args.output_format} -i {str(fastaFile.absolute().resolve())} -o {str((output_dir / postfix).absolute().resolve())} -a {str(match.absolute().resolve())}\n")    
            
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
        default=['mBed', 'clustal', 'muscle', 'esm-43M', 'esm-650M', 'prose_mt_100', 'prose_mt_6K', 'mafft'],
    )
    parser.add_argument(
        "--output_format",
        type=str,
        help="MSA output file format",
        default="msf",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)