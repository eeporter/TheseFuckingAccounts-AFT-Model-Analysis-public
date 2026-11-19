#!/usr/bin/env bash
# fastscan2 speed + equality check against fastscan on the Spark pilot's comment file. Reads fastscan's pairs; changes nothing there.
# Tiny test: MAX_ROW_GROUPS=2 sbatch ... ; whole file: no MAX_ROW_GROUPS.
# Jobs: 43324646 (MAX_ROW_GROUPS=2, tiny check) and 43328586 (whole pilot file: PASS, 427 pairs).
# Steps: build libfastscan2.so; test_clean_v1.py (C cleanup == pyarrow cleanup on all code points + random strings);
#   pilot_fast2_v1.py (pairs equal fastscan pilot job 43321861). Output: pilot_fast2_out_v1/ (ids + scores, summary).
# The .so is built at the default path fastscan2.py loads (fastscan2/libfastscan2.so); this was a single job, before
#   the array runs moved to private per-task build folders.
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} max_row_groups=${MAX_ROW_GROUPS:-all}"; gcc --version | head -1
gcc -O3 -shared -fPIC -o $D/libfastscan2.so $D/fastscan2.c
echo "== cleanup check $(date +%T)"; apptainer exec $B $SIF python3 $D/test_clean_v1.py
echo "== pilot $(date +%T)"; apptainer exec $B $SIF python3 $D/pilot_fast2_v1.py
echo "== done $(date +%T)"
