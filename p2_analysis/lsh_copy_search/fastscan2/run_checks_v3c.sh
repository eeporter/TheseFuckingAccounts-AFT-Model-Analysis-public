#!/usr/bin/env bash
# Extra checks on the merged whole-pool output full_out_v3 (array 43340006 + rerun 43360110), added after the 18 Nov
# code review. Counts only; reads no text into the log.
#   1. time offset: is archive CreatedAt fixed UTC-8 (the search adds 8 h) or Pacific time with daylight saving?
#      Query texts that are also archive rows give an independent clock (created_utc); offsets are tabulated by month.
#   2. invariants v3: all v2 per-row rules + no query matched to its own archive row + rows on disk == summed pairs.
#   3. brute force v2: ALL 3.3M queries against a random slice of archive rows (50,000 from 2 comment + 1 post random
#      row groups, plus 2,000 from one row group with > 2M pairs), counting every shingle overlap; output pairs are
#      collected for every sampled row, including rows that fail a rule.
# Submit: sbatch -A comdata -p cpu-g2 -c 4 --mem 96G --time 12:00:00 -J checks_v3c \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/checks_v3c_%j.out run_checks_v3c.sh
# Job: submitted as 43361281 (checks_v3c). Step 1 found the archive time is Pacific local time with daylight saving
#   (227,893 rows: 61% 8 h, 39% 7 h), which led to rules4.py and the v4 rerun.
# Inputs: full_out_v3/ (steps 2-3), query set lsh_v2 (steps 1, 3), archive files (steps 1, 3).
# Outputs: JSON counts in the job log only. Each step prints its exit code (rc); set -u without -e, so a failing step
#   does not stop the later ones.
set -u
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
B="--bind /mmfs1/gscratch:/gscratch"
L2=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh_v2
# NEW = merged whole-pool output of array 43340006 + rerun 43360110 (old + 8 h time rule)
NEW=$D/full_out_v3
cd $D
echo "== host $(hostname) $(date +%T)"
# step 1: time_offset_check_v1.py on 40 random comment files and 20 random post files (seed 1)
echo "== 1 time offset (CreatedAt vs created_utc, by month)"; LSH_DIR=$L2 apptainer exec $B $SIF python3 $D/time_offset_check_v1.py --ncomment 40 --npost 20 --seed 1; echo "rc=$?"
# step 2: invariants_check_v3.py on NEW with MINCHAR 50
echo "== 2 invariants v3"; apptainer exec $B $SIF python3 $D/invariants_check_v3.py $NEW 50; echo "rc=$?"
# step 3: bruteforce_check_v2.py (rules2, + 8 h, matching how full_out_v3 was made) on NEW, seed 1
echo "== 3 brute force v2"; LSH_DIR=$L2 apptainer exec $B $SIF python3 $D/bruteforce_check_v2.py --out $NEW --rows 50000 --nrg-comment 2 --nrg-post 1 --big-rows 2000 --seed 1; echo "rc=$?"
echo "== done $(date +%T)"
