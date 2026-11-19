#!/usr/bin/env bash
# FULL copy search with fastscan2, all rules, one pass at 50 characters (the 140 set is the rows with both vis_len >= 140).
# v2 (18 Nov 2026): same as run_full_fast2_v1.sh but fast2_full_v2.py with 12 workers + ARROW_CPU1=1 (A/B job 43332090: identical pairs, ~3.8x faster).
# Slurm array: task i of N takes every N-th older-batch file (comments 074ac9d9 + submissions f218b03b; newer batch skipped).
# Each task first checks itself against fastscan's rule code on one comment and one post row group and stops if they differ.
# Submit: sbatch -A comdata -p cpu-g2 --array=0-29 -c 4 --mem 32G --time 2:00:00 -J full_fast2_v2 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/full_fast2_v2_%A_%a.out run_full_fast2_v2.sh
# Job: array 43332591 (second full run, 987 accounts, 12 workers), output full_out_v2/.
# Checked against full_out_v1 by run_compare_full_v1.sh (job 43332603): IDENTICAL, 24,277,320 pairs.
# Each task writes pairs_taskNNNN_pKKKK.parquet files and summary_taskNNNN.json (DONE, rows, pairs, timings).
# --query pre2022: query texts created before 30 Nov 2022; --files older: older batch only (2,000 files).
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
OUT=$D/${OUT_NAME:-full_out_v2}
BLD=$D/build/${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}     # private build per task: array tasks must not share a .so being written
mkdir -p $BLD $OUT
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
# libfastscan2.so = fastscan2 scanner; libfastscan.so = the unchanged fastscan (ref/) used by --selfcheck.
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
gcc -O3 -shared -fPIC -o $BLD/libfastscan.so $D/ref/fastscan.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so FASTSCAN_LIB=$BLD/libfastscan.so
# 12 worker processes on 4 cores, as chosen by A/B job 43332090; ARROW_CPU1=1 keeps each worker to one Arrow thread
POOL_WORKERS=${POOL_WORKERS:-12} ARROW_CPU1=${ARROW_CPU1:-1} apptainer exec $B $SIF python3 $D/fast2_full_v2.py --minchar ${MINCHAR:-50} --query pre2022 --files older --selfcheck --out $OUT
echo "== done $(date +%T)"
