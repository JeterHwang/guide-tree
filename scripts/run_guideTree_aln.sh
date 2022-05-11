export PATH=/b07068/MSA/guide-tree/Benchmarks/programs:$PATH
CPU=1
MEM=8
IN=
OUT=
PROG= 
TREE_IN=
TREE_OUT=
FMT=
function usage()
{
    printf "usage: $0  -p <prog> -f <format> -i <in> -o <out> -a <guide-tree-in> -b <guide-tree-out>\n\n" ;
    printf "Options:\n-t <threads>\n-m <mem>\n\n";
    printf "Valid options for <prog> include:\n   kalign\n   muscle\n   clustal\n\n";
    exit 1;
}

while getopts t:m:p:f:i:o:a:b:  opt
do
    case ${opt} in
        t) CPU=${OPTARG};;
        m) MEM=${OPTARG};;
        p) PROG=${OPTARG};;
        f) FMT=${OPTARG};;
        i) IN=${OPTARG};;
        o) OUT=${OPTARG};;
        a) TREE_IN=${OPTARG};;
        b) TREE_OUT=${OPTARG};;
        *) usage;;
    esac
done

if [ "${PROG}" == "" ]; then usage; fi
if [ "${IN}" == "" ]; then usage; fi
if [ "${OUT}" == "" ]; then usage; fi
if [ "${TREE_IN}" != "" ] && [ "${TREE_OUT}" != "" ]; then usage; fi
if [ "${PROG}" == "mafft" ] && [ "${FMT}" == "msf" ]; then usage; fi

SLURMMEM=$MEM"G"

CMD= 

if [ "${PROG}" == "kalign" ]; then 
    if [ "${FMT}" == "msf" ]; then
        CMD="kalign -i $IN -f msf -o $OUT"
    else
        CMD="kalign -i $IN -o $OUT"
    fi
fi

if [ "${PROG}" == "muscle" ]; then 
    if [ "${TREE_IN}" != "" ]; then
        if [ "${FMT}" == "msf" ]; then
            CMD="muscle3.8.31_i86linux64 -msf -in $IN -out $OUT -usetree_nowarn $TREE_IN"
        else
            CMD="muscle3.8.31_i86linux64 -in $IN -out $OUT -usetree_nowarn $TREE_IN"
        fi
    elif [ "${TREE_OUT}" != "" ]; then
        CMD="muscle3.8.31_i86linux64 -in $IN -out $OUT -tree2 $TREE_OUT"
    else
        if [ "${FMT}" == "msf" ]; then
            CMD="muscle3.8.31_i86linux64 -msf -in $IN -out $OUT"
        else
            CMD="muscle3.8.31_i86linux64 -in $IN -out $OUT"
        fi
    fi
fi

if [ "${PROG}" == "clustal" ]; then 
    if [ "${TREE_IN}" != "" ]; then
        if [ "${FMT}" == "msf" ]; then
            CMD="clustalo --outfmt=msf --in $IN --out $OUT --guidetree-in $TREE_IN --force"
        else
            CMD="clustalo --in $IN --out $OUT --guidetree-in $TREE_IN --force"
        fi
    elif [ "${TREE_OUT}" != "" ]; then
        CMD="clustalo --in $IN --out $OUT --guidetree-out $TREE_OUT --force"
    else
        if [ "${FMT}" == "msf" ]; then
            CMD="clustalo --outfmt=msf --in $IN --out $OUT --force"
        else
            CMD="clustalo --in $IN --out $OUT --force"
        fi
    fi
fi

if [ "${PROG}" == "mafft" ]; then
    if [ "${TREE_IN}" != "" ]; then
        CMD="mafft --treein $TREE_IN --localpair $IN"
    elif [ "${TREE_OUT}" != "" ]; then
        CMD="mafft --retree 0 --treeout --localpair $IN"
    else
        CMD="mafft --localpair $IN"
    fi
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
    if [ "${PROG}" == "mafft" ]; then
        $CMD > $OUT
    else
        $CMD
    fi
fi

