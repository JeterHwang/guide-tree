export PATH=/home/jeter/Desktop/ICS/MSA/Benchmarks/programs:$PATH
# find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/data-set1  /home/jeter/Desktop/ICS/MSA/Benchmarks/data/data-set2  -name "*_test.fa" -exec rm -rf {} \;
printf "Generate commands to reformat the reference alignments\n\n";

find /home/jeter/Desktop/ICS/MSA/Benchmarks/scratch -name "*.fa"  -o -name '*.fasta' | grep mafft |  awk -v outdir="$SCRATCHDIR" '{printf "kalign %s --reformat --rename -f msf -o %s.msf\n", $1,$1 }' > run_reformat.sh
chmod 755 run_reformat.sh
./run_reformat.sh

