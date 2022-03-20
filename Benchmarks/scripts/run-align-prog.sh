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

printf "Generating balibase run commands\n";

find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/bb3_release -name "*.tfa" |\
    awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/);\
                                    printf "./run_aln.sh -p kalign -i %s -f msf -o %s/%s_%s_kalign.msf\n", $1,outdir,a[n-2],a[n-1];\
                                    printf "./run_aln.sh -p muscle -i %s -f msf -o %s/%s_%s_muscle.msf\n", $1,outdir,a[n-2],a[n-1];\
                                    printf "./run_aln.sh -p clustal -i %s -f msf -o %s/%s_%s_clustal.msf\n", $1,outdir,a[n-2],a[n-1];\
                                    printf "./run_aln.sh -p mafft -i %s -f msf -o %s/%s_%s_mafft.fa\n", $1,outdir,a[n-2],a[n-1];\
                         }' > run_benchmark.sh

# printf "Generating bralibase run commands\n";

# find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/data-set1  /home/jeter/Desktop/ICS/MSA/Benchmarks/data/data-set2  -name "*_test.fa" |\
#     grep unaligned |\
#     awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/);\
#                                   printf "./run_aln.sh -p kalign -i %s -f msf -o %s/%s_%s_%s_kalign.msf\n", $1,outdir,a[n-4],a[n-3],a[n-2];\
#                                   printf "./run_aln.sh -p kalign2 -i %s -f msf -o %s/%s_%s_%s_kalign2.msf\n", $1,outdir,a[n-4],a[n-3],a[n-2];\
#                                   printf "./run_aln.sh -p muscle -i %s -f msf -o %s/%s_%s_%s_muscle.msf\n", $1,outdir,a[n-4],a[n-3],a[n-2];\
#                                   printf "./run_aln.sh -p clustal -i %s -f msf -o %s/%s_%s_%s_clustal.msf\n", $1,outdir,a[n-4],a[n-3],a[n-2];\
#            }' >> run_benchmark.sh

# printf "Generating Quantest2 run commands\n";

# find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/QuanTest2/Test -name "*.vie" |\
#     awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/);\
#               printf "./run_aln.sh -p kalign -i %s -o %s/%s_kalign.afa\n", $1,outdir,a[n-1];\
#               printf "./run_aln.sh -p kalign2 -i %s -o %s/%s_kalign2.afa\n", $1,outdir,a[n-1];\
#               printf "./run_aln.sh -p muscle -i %s -o %s/%s_muscle.afa\n", $1,outdir,a[n-1];\
#               printf "./run_aln.sh -p clustal -i %s -o %s/%s_clustal.afa\n", $1,outdir,a[n-1];\
# }' >>  run_benchmark.sh

chmod 755 run_benchmark.sh
