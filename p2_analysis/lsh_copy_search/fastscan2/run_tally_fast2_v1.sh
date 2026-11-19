#!/usr/bin/env bash
# Per-text tally of the full copy search. Reads full_out_v1 and lsh/ only; writes fastscan2/tally_out_v1/.
# Submit: sbatch -A comdata -p cpu-g2 -c 4 --mem 48G --time 0:30:00 -J tally_fast2_v1 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/tally_fast2_v1_%j.out run_tally_fast2_v1.sh
# Job: 43331456. Uses tally_fast2_v1.py. Outputs: tally_out_v1/searched_texts_v1.parquet, per_text_tally_v1.parquet,
#   tally_summary_v1.json (counts). Runs after the first full run (array 43329112).
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
# private build folder; libfastscan2.so is needed because rules2.py imports fastscan2, which loads it on import
BLD=$D/build/tally_${SLURM_JOB_ID:-x}; mkdir -p $BLD
cd $D
echo "== host $(hostname) $(date +%T)"
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so
apptainer exec $B $SIF python3 $D/tally_fast2_v1.py --pairs $D/full_out_v1 --lsh /gscratch/comdata/users/eeporter/reddit_copy_check/lsh --out $D/tally_out_v1
echo "== done $(date +%T)"
