#!/usr/bin/env bash
# Sample page of matched pairs (PRIVATE: holds comment text). Reads full_out_v1, lsh/ and a few archive files; writes fastscan2/review_v1/.
# Submit: sbatch -A comdata -p cpu-g2 -c 4 --mem 48G --time 0:30:00 -J sample_page_v1 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/sample_page_v1_%j.out run_sample_page_v1.sh
# Job: 43331457. Uses sample_page_v1.py. Outputs: review_v1/sample_matches_v1.html (private) and
#   review_v1/sample_summary_v1.json (counts only).
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
# private build folder; libfastscan2.so is needed because rules2.py imports fastscan2, which loads it on import
BLD=$D/build/sample_${SLURM_JOB_ID:-x}; mkdir -p $BLD
cd $D
echo "== host $(hostname) $(date +%T)"
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so
apptainer exec $B $SIF python3 $D/sample_page_v1.py --pairs $D/full_out_v1 --lsh /gscratch/comdata/users/eeporter/reddit_copy_check/lsh --arch /gscratch/comdata/output --out $D/review_v1
# owner-only permissions on the page with texts
chmod 600 $D/review_v1/sample_matches_v1.html
echo "== done $(date +%T)"
