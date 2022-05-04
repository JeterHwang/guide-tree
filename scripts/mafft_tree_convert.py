from pathlib import Path
from argparse import ArgumentParser, Namespace

def main(args):
    for i, treeFile in enumerate(list(args.tree_dir.glob('**/*.tree'))):
        newlines = []
        with open(treeFile, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[0] not in ['(', ':']:
                    underline = line.find('_')
                    newlines.append(line[underline+1 : ])
                else:
                    newlines.append(line)    
        
        treeFile.unlink()
        
        with open(treeFile, 'w') as f:
            for line in newlines:
                f.write(line)
            
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--tree_dir",
        type=Path,
        help="Path to generated trees",
        default="../trees/bb3_release/mafft",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)