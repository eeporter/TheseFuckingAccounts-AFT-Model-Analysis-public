#!/usr/bin/env bash
# Checks the second full run (full_out_v2) against the first (full_out_v1, job 43329112): identical pairs? Counts only.
# Submit after the full run: sbatch --dependency=afterok:<array job id> -A comdata -p cpu-g2 -c 4 --mem 32G --time 1:00:00 \
#   -J compare_full_v1 -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/compare_full_v1_%j.out run_compare_full_v1.sh
# Job: 43332603: IDENTICAL, 24,277,320 pairs (full_out_v2 = array 43332591). Uses compare_full_v1.py, which
# compares the two folders as multisets of 64-bit row hashes and checks both runs are complete (2,000 files).
# afterok on the array: runs only if every task of the second full run succeeded. Output: job log only.
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
echo "== host $(hostname) $(date +%T)"
apptainer exec $B $SIF python3 $D/compare_full_v1.py $D/full_out_v1 $D/full_out_v2
echo "== done $(date +%T)"
