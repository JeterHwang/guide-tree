SCRATCHDIR = "/home/b07068/MSA/guide-tree/Benchmarks/scratch"
printf "Generating kalign scoring run script\n";
find /home/ntu329/jeter/alignment/Benchmarks/data/bb3_release -name "*.xml" | awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/); ;printf "/home/ntu329/jeter/alignment/Benchmarks/programs/bali_score %s %s/%s_%s_kalign.msf | grep auto\n", $1,outdir,a[n-2],a[n-1] }' > score_bb3_kalign.sh
chmod 755  score_bb3_kalign.sh

printf "Generating muscle scoring run script\n";
find /home/ntu329/jeter/alignment/Benchmarks/data/bb3_release -name "*.xml" | awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/); ;printf "/home/ntu329/jeter/alignment/Benchmarks/programs/bali_score %s %s/%s_%s_muscle.msf | grep auto\n", $1,outdir,a[n-2],a[n-1] }' > score_bb3_muscle.sh
chmod 755  score_bb3_muscle.sh

printf "Generating clustal scoring run script\n";
find /home/ntu329/jeter/alignment/Benchmarks/data/bb3_release -name "*.xml" | awk -v outdir="$SCRATCHDIR" '{n=split ($1,a,/[\/,.]/); ;printf "/home/ntu329/jeter/alignment/Benchmarks/programs/bali_score %s %s/%s_%s_clustal.msf | grep auto\n", $1,outdir,a[n-2],a[n-1] }' > score_bb3_clustal.sh
chmod 755  score_bb3_clustal.sh

./score_bb3_kalign.sh > scores_balibase_kalign.csv
./score_bb3_kalign2.sh > scores_balibase_kalign2.csv
./score_bb3_muscle.sh > scores_balibase_muscle.csv
./score_bb3_clustal.sh > scores_balibase_clustal.csv