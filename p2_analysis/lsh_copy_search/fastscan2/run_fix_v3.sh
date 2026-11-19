#!/usr/bin/env bash
# Rerun the 3 failed tasks (1, 3, 22) of full_v3 (array 43340006) with fast2_full_v3.py: identical matching, but row
# groups with more than 2M pairs are built and written in slices by the worker (fixes the Arrow 2 GB string overflow and
# the out-of-memory), and a killed worker makes the task fail at once instead of hanging.
# --ntask 30 is passed explicitly so each task gets exactly the same files and index order as in the 30-task run.
# Writes to a NEW folder full_out_v3_fix/ (full_out_v3/ untouched until run_merge_v3.sh).
# Submit: sbatch -A comdata -p cpu-g2 --array=1,3,22 -c 4 --mem 200G --time 4:00:00 -J fix_v3 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/fix_v3_%A_%a.out run_fix_v3.sh
# Job: array 43360110 (tasks 1, 3, 22). Then run_merge_v3.sh (43360111) and run_checks_v3b.sh (43360112).
# Why --ntask 30: fast2_full_v3.py takes files[task::ntask] and builds its index order from the first file of that
#   slice. Without --ntask, ntask defaults to SLURM_ARRAY_TASK_COUNT, which is 3 for --array=1,3,22, so the tasks would
#   get the wrong files. With --ntask 30 task t gets every 30th file starting at t, as in array 43340006.
# --chunk 2000000: pairs of a row group are built in slices of at most 2M, so no Arrow string column passes 2 GB
#   (the "Negative offsets" crash); row groups with more pairs are written by the worker itself (pw<pid> files).
# 200G memory (vs 48G in run_full_v3.sh): these tasks hit the out-of-memory in array 43340006.
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
L2=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh_v2
OUT=$D/full_out_v3_fix
# Private build folder per job/task: the C code is compiled here, outside the container, and the Python modules are
# pointed at it through FASTSCAN2_LIB / FASTSCAN_LIB. Concurrent jobs or array tasks must not share one .so that
# another task may be rewriting while it is loaded.
BLD=$D/build/${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $BLD $OUT
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} task=${SLURM_ARRAY_TASK_ID} of 30"
# libfastscan2.so = fastscan2 scanner; libfastscan.so = the unchanged fastscan (ref/) used by --selfcheck.
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
gcc -O3 -shared -fPIC -o $BLD/libfastscan.so $D/ref/fastscan.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so FASTSCAN_LIB=$BLD/libfastscan.so
# same settings as run_full_v3.sh (lsh_v2 queries, 12 workers, one Arrow thread each, self-check), plus --chunk
LSH_DIR=$L2 POOL_WORKERS=12 ARROW_CPU1=1 apptainer exec $B $SIF python3 $D/fast2_full_v3.py --minchar 50 --query pre2022 --files older \
  --task ${SLURM_ARRAY_TASK_ID} --ntask 30 --selfcheck --chunk 2000000 --out $OUT
echo "== done $(date +%T)"
