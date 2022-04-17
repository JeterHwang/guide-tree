export PATH=/home/ntu329/jeter/alignment/Benchmarks/programs:$PATH
CPU=1
MEM=8
IN=
OUT=
PROG= 
TREE_IN=
TREE_OUT=
function usage()
{
    printf "usage: $0  -p <prog> -i <in> -o <out> -ti <guide-tree-in> -to <guide-tree-out>\n\n" ;
    printf "Options:\n-t <threads>\n-m <mem>\n\n";
    printf "Valid options for <prog> include:\n   kalign\n   muscle\n   clustal\n\n";
    exit 1;
}

while getopts t:m:p:i:o:f:ti:to  opt
do
    case ${opt} in
        t) CPU=${OPTARG};;
        m) MEM=${OPTARG};;
        p) PROG=${OPTARG};;
        i) IN=${OPTARG};;
        o) OUT=${OPTARG};;
        ti) TREE_IN=${OPTARG};;
        to) TREE_OUT=${OPTARG};;
        *) usage;;
    esac
done

if [ "${PROG}" == "" ]; then usage; fi
if [ "${IN}" == "" ]; then usage; fi
if [ "${OUT}" == "" ]; then usage; fi
if [ "${TREE_IN}" != ""] && [ "${TREE_OUT}" != ""]; then usage; fi

SLURMMEM=$MEM"G"

CMD= 

if [ "${PROG}" == "kalign" ]; then 
    CMD="kalign -i $IN -f msf -o $OUT"
fi

if [ "${PROG}" == "muscle" ]; then 
    if [ "${TREE_IN}" != ""]; then
        CMD="muscle3.8.31_i86linux32 -msf -in $IN -out $OUT -usetree_nowarn $TREE_IN"
    elif ["${TREE_OUT}" != ""]; then
        CMD="muscle3.8.31_i86linux32 -maketree -in $IN -out $TREE_OUT"
    else
        CMD="muscle3.8.31_i86linux32 -msf -in $IN -out $OUT"
fi

if [ "${PROG}" == "clustal" ]; then 
    if [ "${TREE_IN}" != ""]; then
        CMD="clustalo-1.2.4-Ubuntu-x86_64 --outfmt=msf --in $IN --out $OUT --guidetree-in $TREE_IN"
    elif ["${TREE_OUT}" != ""]; then
        CMD="muscle3.8.31_i86linux32 --in $IN --out $OUT --guidetree-out $TREE_OUT"
    else
        CMD="clustalo-1.2.4-Ubuntu-x86_64 --outfmt=msf --in $IN --out $OUT"
fi

if [ "${PROG}" == "mafft" ]; then
    if [ "${TREE_IN}" != "" ]; then
        CMD="mafft --localpair --treein $TREE_IN $IN > $OUT"
    elif [ "${TREE_OUT}" != "" ]; then
        CMD="mafft --localpair --treeout $TREE_OUT $IN > $OUT"
    else
        CMD="mafft --localpair $IN > $OUT"
fi

HAS_SLURM=0
#    printf "Running Sanity checks:\n";

if which sbatch >/dev/null; then
    HAS_SLURM=1
fi

#     echo $HAS_SLURM
if [ $HAS_SLURM = 1 ]; then 
    echo "YES"
    sbatch <<EOT
#!/usr/bin/env bash

#SBATCH --cpus-per-task=$CPU
#SBATCH --mem=$SLURMMEM
#SBATCH -t 10-12:30 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

$CMD
exit 0
EOT
else 
    $CMD 
fi

