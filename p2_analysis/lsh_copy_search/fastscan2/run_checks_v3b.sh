#!/usr/bin/env bash
# The 5 approved checks on the merged full_out_v3 (counts only). Same as run_checks_v3.sh except invariants uses the
# streaming invariants_check_v2.py (v1 ran out of memory at ~5B rows).
# Submit: sbatch --dependency=afterok:<merge_v3 job id> -A comdata -p cpu-g2 -c 4 --mem 64G --time 12:00:00 -J checks_v3b \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/checks_v3b_%j.out run_checks_v3b.sh
# Job: 43360112, on full_out_v3 after run_merge_v3.sh (job 43360111) put in the rerun tasks 1, 3, 22 (array 43360110).
# Checks: 1+3 completeness and self-checks; 2 regression (compare_subset_v1.py, old queries vs full_out_v2);
#   4 invariants (invariants_check_v2.py, streaming); 5 brute force (bruteforce_check_v1.py). Output: job log only.
# afterok on the merge job: runs only if the merge succeeded (the merge itself stops unless the reruns are complete).
# set -u without -e: every check runs and prints its own exit code (rc=).
set -u
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
# L2 = whole-pool query set; L1 = the old query set of full_out_v2 (used by the regression check)
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
         cpu_h=round(sum(s.get('cpu_s', 0) for s in S) / 3600, 1), selfcheck_n=len(sc), selfcheck_all_ok=bool(sc) and all(sc))
r['complete'] = r['tasks'] == 30 and done == 30 and files == 2000 and rows == 23604630484
print(json.dumps(r))
PY
echo "rc=$?"
echo "== 2 regression (old 71,420 queries vs full_out_v2)"; apptainer exec $B $SIF python3 $D/compare_subset_v1.py $OLD $NEW $L1; echo "rc=$?"
echo "== 4 invariants (streaming)"; apptainer exec $B $SIF python3 $D/invariants_check_v2.py $NEW 50; echo "rc=$?"
echo "== 5 brute force"; LSH_DIR=$L2 apptainer exec $B $SIF python3 $D/bruteforce_check_v1.py --out $NEW --nrg-comment 2 --nrg-post 1 --nq 1000 --seed 1; echo "rc=$?"
echo "== done $(date +%T)"
