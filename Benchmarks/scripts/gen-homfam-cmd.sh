SCRATCHDIR="/home/jeter/Desktop/ICS/MSA/Benchmarks/scratch"
DIR=`pwd`
function usage()
{
    printf "This script will generate scripts to run kalign, clustal omega and muscle on the balibase MSA benchmark data set.\n\n" ;
    printf "usage: $0\n\n" ;
    exit 1;
}

while getopts h  opt
do
    case ${opt} in
        h) usage;;
        *) usage;;
    esac
done

printf "export PATH=/home/jeter/Desktop/ICS/MSA/Benchmarks/programs:$PATH\n\n" > run_homfam.sh
find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/homfam  -name "*_testaln.fa" | awk -v outdir="$SCRATCHDIR" '{
ref_aln=$1;gsub("_testaln.fa", "_ref.msf",ref_aln);
printf "./run_timealn.sh -p kalign -i %s -r %s\n", $1,ref_aln;
printf "./run_timealn.sh -p muscle -i %s -r %s\n", $1,ref_aln;
printf "./run_timealn.sh -p clustal -i %s -r %s\n", $1,ref_aln;
printf "./run_timealn.sh -p mafft -i %s -r %s\n", $1,ref_aln;
             }' >> run_homfam.sh

chmod 755 run_homfam.sh