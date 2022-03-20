SCRATCHDIR="/home/jeter/Desktop/ICS/MSA/Benchmarks/scratch"

printf "Generating kalign scoring run script\n";
find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/bb3_release -name "*.xml" | awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/); ;printf "/home/jeter/Desktop/ICS/MSA/Benchmarks/programs/bali_score %s %s/%s_%s_kalign.msf | grep auto\n", $1,outdir,a[n-2],a[n-1] }' > score_bb3_kalign.sh
chmod 755  score_bb3_kalign.sh

printf "Generating mafft scoring run script\n";
find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/bb3_release -name "*.xml" | awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/); ;printf "/home/jeter/Desktop/ICS/MSA/Benchmarks/programs/bali_score %s %s/%s_%s_mafft.fa.msf | grep auto\n", $1,outdir,a[n-2],a[n-1] }' > score_bb3_mafft.sh
chmod 755  score_bb3_mafft.sh

printf "Generating muscle scoring run script\n";
find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/bb3_release -name "*.xml" | awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/); ;printf "/home/jeter/Desktop/ICS/MSA/Benchmarks/programs/bali_score %s %s/%s_%s_muscle.msf | grep auto\n", $1,outdir,a[n-2],a[n-1] }' > score_bb3_muscle.sh
chmod 755  score_bb3_muscle.sh

printf "Generating clustal scoring run script\n";
find /home/jeter/Desktop/ICS/MSA/Benchmarks/data/bb3_release -name "*.xml" | awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/); ;printf "/home/jeter/Desktop/ICS/MSA/Benchmarks/programs/bali_score %s %s/%s_%s_clustal.msf | grep auto\n", $1,outdir,a[n-2],a[n-1] }' > score_bb3_clustal.sh
chmod 755  score_bb3_clustal.sh

./score_bb3_kalign.sh > scores_balibase_kalign.csv
./score_bb3_mafft.sh > scores_balibase_mafft.csv
./score_bb3_muscle.sh > scores_balibase_muscle.csv
./score_bb3_clustal.sh > scores_balibase_clustal.csv