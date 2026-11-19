#!/usr/bin/env bash
# The 5 approved checks on full_out_v3 (counts only). Runs after the whole array, even if some task failed (afterany).
# Submit: sbatch --dependency=afterany:<full_v3 array job id> -A comdata -p cpu-g2 -c 4 --mem 64G --time 8:00:00 -J checks_v3 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/checks_v3_%j.out run_checks_v3.sh
# Job: 43340007, on full_out_v3 of the whole-pool run (array 43340006, run_full_v3.sh).
# Checks: 1+3 completeness (30 tasks DONE, 2,000 files, 23,604,630,484 rows) and every task self-check ok;
#   2 regression: compare_subset_v1.py, old 71,420 queries (lsh/) must give exactly the full_out_v2 pairs;
#   4 invariants: invariants_check_v1.py on every row (minchar 50);
#   5 brute force: bruteforce_check_v1.py on 2 comment + 1 post row groups, 1,000 random queries (lsh_v2), seed 1.
# Output: the job log only (counts and one rc= line per check).
# afterany (not afterok) on the array: the checks run even if some tasks failed, so failures show up as counts here.
# set -u without -e: a failing check does not stop the later checks; each prints its own exit code (rc=).
set -u
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
# L2 = whole-pool query set (38,106 accounts); L1 = the old query set (987 accounts) of full_out_v1/v2
L2=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh_v2
L1=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh
NEW=$D/full_out_v3
OLD=$D/full_out_v2
cd $D
echo "== host $(hostname) $(date +%T)"
echo "== 1+3 completeness and built-in self-check"
apptainer exec $B $SIF python3 - $NEW <<'PY'
import json, glob, sys
S = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + '/summary_task*.json'))]
rows = sum(s.get('rows', 0) for s in S); files = sum(s.get('files_this_task', 0) for s in S)
done = sum(1 for s in S if s.get('DONE')); sc = [x['ok'] for s in S for x in s.get('selfcheck', [])]
r = dict(tasks=len(S), tasks_done=done, files=files, rows=rows, rows_expected=23604630484,
         pairs=sum(s.get('pairs', 0) for s in S), queries=S[0].get('queries') if S else None,
         setup_s_max=max((s.get('setup_s', 0) for s in S), default=None), wall_s_max=max((s.get('wall_s', 0) for s in S), default=None),
         cpu_h=round(sum(s.get('cpu_s', 0) for s in S) / 3600, 1), max_rss_one_worker_GB=max((s.get('max_rss_one_worker_GB') or 0 for s in S), default=None),
         selfcheck_n=len(sc), selfcheck_all_ok=bool(sc) and all(sc))
r['complete'] = r['tasks'] == 30 and done == 30 and files == 2000 and rows == 23604630484
print(json.dumps(r))
PY
echo "rc=$?"
echo "== 2 regression (old 71,420 queries vs full_out_v2)"; apptainer exec $B $SIF python3 $D/compare_subset_v1.py $OLD $NEW $L1; echo "rc=$?"
echo "== 4 invariants"; apptainer exec $B $SIF python3 $D/invariants_check_v1.py $NEW 50; echo "rc=$?"
echo "== 5 brute force"; LSH_DIR=$L2 apptainer exec $B $SIF python3 $D/bruteforce_check_v1.py --out $NEW --nrg-comment 2 --nrg-post 1 --nq 1000 --seed 1; echo "rc=$?"
echo "== done $(date +%T)"
