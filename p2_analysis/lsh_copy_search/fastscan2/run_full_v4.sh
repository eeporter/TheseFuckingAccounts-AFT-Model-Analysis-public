#!/usr/bin/env bash
# FULL copy search, whole-pool queries (lsh_v2, 3.3M queries at minchar 50), with the daylight-saving time fix:
# fast2_full_v4.py = fast2_full_v3.py (chunked writes, fail-fast pool) using rules4.py, where the archive CreatedAt is
# read as America/Los_Angeles local time instead of + 8 h. Everything else is as in run_full_v3.sh (job 43340006).
# Writes to a NEW folder full_out_v4/ (full_out_v3/ is kept for the old-vs-new comparison).
# --ntask is passed explicitly (30) so file assignment never depends on how the array is submitted.
# Submit: sbatch -A comdata -p cpu-g2 --array=0-29 -c 4 --mem 64G --time 6:00:00 -J full_v4 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/full_v4_%A_%a.out run_full_v4.sh
# Job: submitted as array 43362084 (job name full_v4, tasks 0-29); checked by checks_v4 43362085 (run_checks_v4.sh).
# Inputs: archive files under ARCH_DIR (default in fast2_full_v4.py: /gscratch/comdata/output, older batch only),
#   query side from lsh_v2 (candidates_v1.parquet, sample_accounts_dates_v1.parquet), code in $D.
# Outputs (in $OUT = full_out_v4/): pairs_taskNNNN_pKKKK.parquet, worker-written pairs_taskNNNN_pw<pid>_KKKKK.parquet
#   for row groups with more than --chunk pairs, and summary_taskNNNN.json (with "DONE": true when the task finished).
#   Per-task build folder $BLD holds the compiled C libraries.
set -eu
# D = code and output folder, SIF = container image, B = bind mount so the container sees /gscratch, L2 = whole-pool
# query set (lsh_v2), OUT = output folder, BLD = per-task build folder (array job id + task id)
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
B="--bind /mmfs1/gscratch:/gscratch"
L2=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh_v2
OUT=$D/full_out_v4
BLD=$D/build/${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $BLD $OUT
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} task=${SLURM_ARRAY_TASK_ID:-0} of 30"
# compile the fastscan2 scanner (Bloom filter + inverted index + exact Jaccard) and the reference fastscan library
# (ref/, used only by --selfcheck) for this node; the Python modules find them through the two env variables
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
gcc -O3 -shared -fPIC -o $BLD/libfastscan.so $D/ref/fastscan.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so FASTSCAN_LIB=$BLD/libfastscan.so
# the scan: 12 worker processes, one Arrow thread each; query texts created before 30 Nov 2022 with >= 50 visible
# characters; this task's every-30th file of the older batch; --selfcheck compares fastscan2 with fastscan's pipeline
# on one comment and one post row group first and stops the task if they differ
LSH_DIR=$L2 POOL_WORKERS=12 ARROW_CPU1=1 apptainer exec $B $SIF python3 $D/fast2_full_v4.py --minchar 50 --query pre2022 --files older \
  --task ${SLURM_ARRAY_TASK_ID:-0} --ntask 30 --selfcheck --chunk 2000000 --out $OUT
echo "== done $(date +%T)"
