#!/usr/bin/env bash
# One-file test of fast_full_v1.py with all current rules: first older-batch comment file, at 140 and at 50 characters.
# Job: no recorded Slurm job in the README.
# Inputs: fastscan.c (compiled here), fast_full_v1.py, the CDSC container, and one archive comment file (the first
# file whose name contains 074ac9d9, the older-batch comment uuid). Outputs: test_out_v1/mc140 and test_out_v1/mc50,
# each with pairs_task0000.parquet (if any pairs) and summary_task0000.json.
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
B="--bind /mmfs1/gscratch:/gscratch"   # /gscratch is a symlink on the host; bind it so the container sees it
F=$(ls /gscratch/comdata/output/reddit_comments_by_author.parquet/*074ac9d9*.parquet | head -1)
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} file=$(basename $F)"
# compile the C core on the compute node with the system gcc (outside the container)
gcc -O3 -Wall -shared -fPIC -o $D/libfastscan.so $D/fastscan.c
# run the full-rules scan on that one file twice: minimum 140 and minimum 50 visible characters
for MC in 140 50; do
  echo "== minchar $MC $(date +%T)"
  apptainer exec $B $SIF python3 $D/fast_full_v1.py --minchar $MC --query pre2022 --files $F --out $D/test_out_v1/mc$MC
done
echo "== done $(date +%T)"
