#!/usr/bin/env bash
# Checks on full_out_v4 (daylight-saving fix), counts only. Runs after the whole array (afterany).
#   1. completeness + built-in self-check (30/30 tasks, 2,000 files, 23,604,630,484 rows, all self-checks ok);
#   2. old vs new (compare_dst_v1): every pair of full_out_v3 is in full_out_v4 (with a_created_utc corrected), and
#      every extra pair is a daylight-saving archive item 0-1 h before the query;
#   3. invariants v3 (per-row rules, no self-match, rows on disk == summed pairs);
#   4. brute force v3 (all queries against a random archive slice, daylight-saving aware time).
# Submit: sbatch --dependency=afterany:<full_v4 array id> -A comdata -p cpu-g2 -c 8 --mem 128G --time 24:00:00 -J checks_v4 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/checks_v4_%j.out run_checks_v4.sh
# Job: submitted as 43362085 (checks_v4), after the full_v4 array 43362084.
# Inputs: full_out_v4/ (new run, Pacific-time rule) and full_out_v3/ (old run, + 8 h), query set lsh_v2, archive files
#   (step 4 only). Outputs: counts and JSON in the job log only; no files are kept. Each step prints its exit code (rc);
#   set -u without -e, so a failing step does not stop the later ones.
set -u
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
B="--bind /mmfs1/gscratch:/gscratch"
L2=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh_v2
# NEW = output of the daylight-saving run (array 43362084), OLD = merged output of the + 8 h runs (43340006 + 43360110)
NEW=$D/full_out_v4
OLD=$D/full_out_v3
cd $D
echo "== host $(hostname) $(date +%T)"
# step 1 (inline Python in the heredoc below): sums the summary_task*.json files of NEW (tasks, DONE flags, files,
# rows, pairs, self-check results); complete = 30 tasks done, 2,000 files, 23,604,630,484 rows
echo "== 1 completeness and built-in self-check"
apptainer exec $B $SIF python3 - $NEW <<'PY'
import json, glob, sys
S = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + '/summary_task*.json'))]
rows = sum(s.get('rows', 0) for s in S); files = sum(s.get('files_this_task', 0) for s in S)
done = sum(1 for s in S if s.get('DONE')); sc = [x['ok'] for s in S for x in s.get('selfcheck', [])]
r = dict(tasks=len(S), tasks_done=done, files=files, rows=rows, rows_expected=23604630484,
         pairs=sum(s.get('pairs', 0) for s in S), queries=S[0].get('queries') if S else None,
         wall_s_max=max((s.get('wall_s', 0) for s in S), default=None), cpu_h=round(sum(s.get('cpu_s', 0) for s in S) / 3600, 1),
         selfcheck_n=len(sc), selfcheck_all_ok=bool(sc) and all(sc))
r['complete'] = r['tasks'] == 30 and done == 30 and files == 2000 and rows == 23604630484
print(json.dumps(r))
PY
echo "rc=$?"
# step 2: compare_dst_v1.py OLD NEW 8 (8 worker processes): old pairs all in new (hash-sum multiset test on the
# pair plus its old-rule archive time) and every extra new pair a daylight-saving archive item 0-1 h before the query
echo "== 2 old (full_out_v3, +8 h) vs new (full_out_v4, Pacific time)"; apptainer exec $B $SIF python3 $D/compare_dst_v1.py $OLD $NEW 8; echo "rc=$?"
# step 3: invariants_check_v3.py on NEW with MINCHAR 50 (per-row rules, no self-match, rows on disk == summed pairs)
echo "== 3 invariants v3"; apptainer exec $B $SIF python3 $D/invariants_check_v3.py $NEW 50; echo "rc=$?"
# step 4: bruteforce_check_v3.py (rules4 time rule): all queries against 50,000 random rows of 2 comment + 1 post row
# groups plus 2,000 rows of one big row group (seed 2); must equal NEW's pairs for those rows
echo "== 4 brute force v3"; LSH_DIR=$L2 apptainer exec $B $SIF python3 $D/bruteforce_check_v3.py --out $NEW --rows 50000 --nrg-comment 2 --nrg-post 1 --big-rows 2000 --seed 2; echo "rc=$?"
echo "== done $(date +%T)"
