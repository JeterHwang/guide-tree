from pathlib import Path
from argparse import ArgumentParser, Namespace

def main(args):
    with open(args.bash_path, 'w') as f:
        for i, xmlFile in enumerate(list(args.xml_dir.glob('**/*.xml'))):
            
            name = xmlFile.stem

            msf_match = list(args.msf_dir.glob(f"*{name}.msf"))
            assert len(msf_match) == 1
            msf_match = msf_match[0]
            
            f.write(f"{str(args.prog_path.absolute().resolve())} {str(xmlFile.absolute().resolve())} {str(msf_match.absolute().resolve())} | grep auto\n")

            
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--prog_path",
        type=Path,
        help="Path to scoring programs",
        default="bali_score",
    )
    parser.add_argument(
        "--xml_dir",
        type=Path,
        help="Path to xml files",
        default="../data/bb3_release",
    )
    ############## NEED TO MODIFY ###############
    parser.add_argument(                        #
        "--msf_dir",                            #
        type=Path,                              #
        help="Path to msf result",              #
        default="../msf/bb3_release/python",    #
    )                                           #
    parser.add_argument(                        #
        "--bash_path",                          #
        type=Path,                              #
        help="Path to run_alignment.sh",        #
        default="score_bb3_python.sh",          #
    )                                           #
    #############################################
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)