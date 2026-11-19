#!/usr/bin/env bash
# FULL copy search with fastscan2, all rules, one pass at 50 characters (the 140 set is the rows with both vis_len >= 140).
# Slurm array: task i of N takes every N-th older-batch file (comments 074ac9d9 + submissions f218b03b; newer batch skipped).
# Each task first checks itself against fastscan's rule code on one comment and one post row group and stops if they differ.
# Submit: sbatch -A comdata -p cpu-g2 --array=0-29 -c 4 --mem 32G --time 2:00:00 -J full_fast2_v1 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/full_fast2_v1_%A_%a.out run_full_fast2_v1.sh
# Job: array 43329112 (first full run, 987-account query set lsh/, 30 tasks), output full_out_v1/.
# Uses fast2_full_v1.py. Followed by tally (43331456), sample page (43331457) and compare_full_v1 (43332603).
# Each task writes pairs_taskNNNN_pKKKK.parquet files and summary_taskNNNN.json (DONE, rows, pairs, timings).
# --query pre2022: query texts created before 30 Nov 2022; --files older: older batch only (2,000 files).
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
OUT=$D/${OUT_NAME:-full_out_v1}
BLD=$D/build/${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}     # private build per task: array tasks must not share a .so being written
mkdir -p $BLD $OUT
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
# libfastscan2.so = fastscan2 scanner; libfastscan.so = the unchanged fastscan (ref/) used by --selfcheck.
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
gcc -O3 -shared -fPIC -o $BLD/libfastscan.so $D/ref/fastscan.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so FASTSCAN_LIB=$BLD/libfastscan.so
apptainer exec $B $SIF python3 $D/fast2_full_v1.py --minchar ${MINCHAR:-50} --query pre2022 --files older --selfcheck --out $OUT
echo "== done $(date +%T)"
