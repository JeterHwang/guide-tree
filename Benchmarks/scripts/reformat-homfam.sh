
export PATH=/home/ntu329/jeter/alignment/Benchmarks/programs:$PATH
DIR=`pwd`
function usage()
{
    printf "This script will merge *_ref.vie and *_test-only.vie alignments and convert them into msf format.\n\n" ;
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
find /home/ntu329/jeter/alignment/Benchmarks/data/homfam  -name "*_test-only.vie" | awk '{\
              test_aln=$1;ref=$1 ;\
              gsub("_test-only.vie", "_ref.vie", ref);\
              gsub("_test-only.vie", "_testaln.fa", test_aln);\
              printf "cat %s %s > tmp.vie\n" ,ref, $1;\
              printf "kalign tmp.vie --reformat --rename -f fasta -o %s\n", test_aln }' > homfam_reformat.sh

find /home/ntu329/jeter/alignment/Benchmarks/data/homfam  -name "*_ref.vie" | awk '{ref=$1;gsub("_ref.vie", "_ref.msf", ref);printf "kalign %s --reformat --rename  -f msf -o %s\n", $1,ref}' >> homfam_reformat.sh

chmod 755 homfam_reformat.sh

./homfam_reformat.sh
