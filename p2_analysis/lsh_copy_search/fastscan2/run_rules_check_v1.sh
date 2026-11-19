#!/usr/bin/env bash
# fastscan2 WITH ALL RULES vs fastscan's rule pipeline, both run here on the same rows: first ROW_GROUPS row groups
# (default 2) of the first older-batch comment file, at 140 and at 50 characters. Reads lsh/ only; changes nothing in fastscan/.
# Job: 43328587. Uses rules_check_v1.py (prints counts, PASS = same pair rows) and test_clean_v1.py.
# Both libraries are built at the default paths the modules load (fastscan2/libfastscan2.so, ref/libfastscan.so);
#   single job, so no other task shares them.
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
# first older-batch comment file (uuid 074ac9d9) in ls order
F=$(ls /gscratch/comdata/output/reddit_comments_by_author.parquet/*074ac9d9*.parquet | head -1)
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} file=$(basename $F) row_groups=${ROW_GROUPS:-2}"
gcc -O3 -shared -fPIC -o $D/libfastscan2.so $D/fastscan2.c
gcc -O3 -shared -fPIC -o $D/ref/libfastscan.so $D/ref/fastscan.c
echo "== cleanup check $(date +%T)"; apptainer exec $B $SIF python3 $D/test_clean_v1.py
for MC in 140 50; do
  echo "== minchar $MC $(date +%T)"
  apptainer exec $B $SIF python3 $D/rules_check_v1.py --file $F --row-groups ${ROW_GROUPS:-2} --minchar $MC --query pre2022
done
echo "== done $(date +%T)"
