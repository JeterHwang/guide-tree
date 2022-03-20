export PATH=/home/ntu329/jeter/alignment/kalign/kalignbenchmark/programs:$PATH
CPU=1
MEM=16
FMT=
IN=
OUT=
PROG= 
function usage()
{
    printf "usage: $0  -p <prog> -i <test aln> -r <ref aln>  \n\n" ;
    printf "Options:\n-f <format>\n-t <threads>\n-m <mem>\n\n";
    printf "Valid options for <prog> include:\n   kalign\n   kalign2\n   muscle\n   clustal\n\n";
    exit 1;
}

while getopts t:m:p:i:r:  opt
do
    case ${opt} in
        t) CPU=${OPTARG};;
        m) MEM=${OPTARG};;
        p) PROG=${OPTARG};;
        i) IN=${OPTARG};;
        r) OUT=${OPTARG};;             
        *) usage;;
    esac
done

if [ "${PROG}" == "" ]; then usage; fi
if [ "${IN}" == "" ]; then usage; fi
if [ "${OUT}" == "" ]; then usage; fi


SLURMMEM=$MEM"G"

CMD= 


CMD="timescorealn -test $IN -ref $OUT -program $PROG --scratch $HOME/kalignbenchmark/scratch -out scores_homfam.csv"         


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

