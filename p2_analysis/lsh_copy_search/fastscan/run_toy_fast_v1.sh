#!/usr/bin/env bash
# Toy test of the exact copy search (fastscan). Builds the C library on the compute node, then runs the toy check
# inside the CDSC container (which has pyarrow + numpy). Reads only the toy files in ../lsh; writes toy_fast_out_v1/.
# Jobs: Hyak Slurm 43301873 (failed before reading data: the container could not see /gscratch, which is a symlink
# on the host; the bind below was added afterwards) and 43301955 (PASS: 33 pairs, same as brute force and Spark).
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
B="--bind /mmfs1/gscratch:/gscratch"   # /gscratch is a symlink on the host; bind it so the container sees it
cd $D
echo "== host $(hostname) $(date +%T)"; gcc --version | head -1
# compile the C core on the compute node with the system gcc (outside the container)
gcc -O3 -Wall -shared -fPIC -o $D/libfastscan.so $D/fastscan.c
# print the container's Python, pyarrow and numpy versions
echo "== container"; apptainer exec $B $SIF python3 -c "import sys, pyarrow, numpy; print(sys.version.split()[0], pyarrow.__version__, numpy.__version__)"
echo "== toy $(date +%T)"; apptainer exec $B $SIF python3 $D/toy_fast_v1.py
echo "== done $(date +%T)"
