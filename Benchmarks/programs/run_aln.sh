export PATH=/home/jeter/Desktop/ICS/MSA/Benchmarks/programs:$PATH
CPU=1
MEM=8
FMT=
IN=
OUT=
PROG= 
function usage()
{
    printf "usage: $0  -p <prog> -i <in> -o <out>  \n\n" ;
    printf "Options:\n-f <format>\n-t <threads>\n-m <mem>\n\n";
    printf "Valid options for <prog> include:\n   kalign\n   muscle\n   clustal\n   mafft\n\n";
    exit 1;
}

while getopts t:m:p:i:o:f:  opt
do
    case ${opt} in
        t) CPU=${OPTARG};;
        m) MEM=${OPTARG};;
        p) PROG=${OPTARG};;
        i) IN=${OPTARG};;
        o) OUT=${OPTARG};;
        f) FMT=${OPTARG};;
        *) usage;;
    esac
done

if [ "${PROG}" == "" ]; then usage; fi
if [ "${IN}" == "" ]; then usage; fi
if [ "${OUT}" == "" ]; then usage; fi


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
    if [ "${FMT}" == "msf" ]; then 
        CMD="muscle3.8.31_i86linux64 -msf -in $IN -out  $OUT"
    else
        CMD="muscle3.8.31_i86linux64 -in $IN -out  $OUT"
    fi
fi

if [ "${PROG}" == "clustal" ]; then 
    if [ "${FMT}" == "msf" ]; then 
        CMD="clustalo-1.2.4-Ubuntu-x86_64 --outfmt=msf --in $IN --out $OUT"
    else
        CMD="clustalo-1.2.4-Ubuntu-x86_64 --outfmt=a2m --in $IN --out $OUT"
    fi
fi

if [ "${PROG}" == "mafft" ]; then 
    CMD="mafft --globalpair $IN > $OUT"
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
    eval $CMD 
fi

