#!/usr/bin/env bash
# FULL copy search, WHOLE-POOL queries (lsh_v2: 38,106 accounts, 8,771,455 texts -> ~3.3M queries at minchar 50).
# Identical to run_full_fast2_v2.sh (job 43332591, validated) except: LSH_DIR=lsh_v2, output full_out_v3/,
# and more generous limits (48G, 12 h) because nothing has run with 3.3M queries yet. Code unchanged.
# Submit: sbatch -A comdata -p cpu-g2 --array=0-29 -c 4 --mem 48G --time 12:00:00 -J full_v3 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/full_v3_%A_%a.out run_full_v3.sh
# Job: array 43340006 (38,106 accounts, 3,318,585 queries). Tasks 1, 3 and 22 failed (Arrow 2 GB string overflow /
# out of memory) and were rerun by run_fix_v3.sh (array 43360110); checks: run_checks_v3.sh (43340007, afterany).
# Output: full_out_v3/ (pairs_taskNNNN_pKKKK.parquet + summary_taskNNNN.json per task).
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
L2=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh_v2
OUT=$D/full_out_v3
# Private build folder per job/task: the C code is compiled here, outside the container, and the Python modules are
# pointed at it through FASTSCAN2_LIB / FASTSCAN_LIB. Concurrent jobs or array tasks must not share one .so that
# another task may be rewriting while it is loaded.
BLD=$D/build/${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $BLD $OUT
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
# libfastscan2.so = fastscan2 scanner; libfastscan.so = the unchanged fastscan (ref/) used by --selfcheck.
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
gcc -O3 -shared -fPIC -o $BLD/libfastscan.so $D/ref/fastscan.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so FASTSCAN_LIB=$BLD/libfastscan.so
# --ntask is not passed: it comes from SLURM_ARRAY_TASK_COUNT (30 for --array=0-29); task t scans every 30th file
LSH_DIR=$L2 POOL_WORKERS=12 ARROW_CPU1=1 apptainer exec $B $SIF python3 $D/fast2_full_v2.py --minchar 50 --query pre2022 --files older --selfcheck --out $OUT
echo "== done $(date +%T)"
