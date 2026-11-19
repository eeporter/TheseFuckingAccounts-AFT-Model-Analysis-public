#!/usr/bin/env bash
# Pilot check of the exact copy search: the Spark pilot's comment file (26.9M rows), pilot rules, 8 workers.
# Job: Hyak Slurm 43321861 (8 cores, 43 s; PASS: all 427 Spark matches found, identical Jaccard, no extra pairs).
# Inputs: fastscan.c (compiled here), pilot_fast_v1.py, the CDSC container. Outputs: pilot_fast_out_v1/ (pairs,
# fast-only pairs, summary.json) written by pilot_fast_v1.py.
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
B="--bind /mmfs1/gscratch:/gscratch"   # /gscratch is a symlink on the host; bind it so the container sees it
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?}"; gcc --version | head -1
# compile the C core on the compute node with the system gcc (outside the container)
gcc -O3 -Wall -shared -fPIC -o $D/libfastscan.so $D/fastscan.c
echo "== pilot $(date +%T)"; apptainer exec $B $SIF python3 $D/pilot_fast_v1.py
echo "== done $(date +%T)"
