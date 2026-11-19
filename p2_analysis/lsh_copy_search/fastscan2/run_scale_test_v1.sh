#!/usr/bin/env bash
# Scaling test: the unchanged copy search (fast2_full_v2.py, 12 workers, minchar 50, pre2022, self-check on) with the
# WHOLE-POOL query set (lsh_v2: 38,106 accounts, 8,771,455 texts -> ~3.3M queries) on the same 2 older-batch comment
# files as A/B run r1_w4_A (sorted positions 620, 621; 34,407,006 rows). Measures setup time, memory, CPU and pairs.
# Then three checks, counts only:
#   regression: output restricted to the 71,420 old queries must equal A/B run r1_w4_A (old queries, same 2 files);
#   invariants on every output row;
#   brute force on 2 random row groups of these 2 files (1,000 random queries + every query with pairs there).
# Submit: sbatch -A comdata -p cpu-g2 -c 4 --mem 64G --time 1:30:00 -J scale_test_v1 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/scale_test_v1_%j.out run_scale_test_v1.sh
# Status: staged, never submitted (no job). The whole-pool run (run_full_v3.sh, array 43340006) was run instead.
# OLD = A/B run r1_w4_A of job 43332090 (old 987-account queries, same two files).
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
L2=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh_v2
L1=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh
OLD=$D/ab_workers_v1/43332090/r1_w4_A
OUT=$D/scale_test_v1/${SLURM_JOB_ID:-x}
# Private build folder per job/task: the C code is compiled here, outside the container, and the Python modules are
# pointed at it through FASTSCAN2_LIB / FASTSCAN_LIB. Concurrent jobs or array tasks must not share one .so that
# another task may be rewriting while it is loaded.
BLD=$D/build/scale_${SLURM_JOB_ID:-x}
mkdir -p $BLD $OUT
cd $D
# the 1,200 older-batch comment files in byte order (LC_ALL=C), so positions 620 and 621 match the A/B test
mapfile -t ALL < <(ls /gscratch/comdata/output/reddit_comments_by_author.parquet/*074ac9d9*.parquet | LC_ALL=C sort)
[ ${#ALL[@]} -eq 1200 ] || { echo "expected 1200 older comment files, found ${#ALL[@]}"; exit 1; }
F="${ALL[620]},${ALL[621]}"
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-x} files 620,621"
# libfastscan2.so = fastscan2 scanner; libfastscan.so = the unchanged fastscan (ref/) used by --selfcheck.
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
gcc -O3 -shared -fPIC -o $BLD/libfastscan.so $D/ref/fastscan.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so FASTSCAN_LIB=$BLD/libfastscan.so
echo "== scan start $(date +%T)"
LSH_DIR=$L2 POOL_WORKERS=12 ARROW_CPU1=1 apptainer exec $B $SIF python3 $D/fast2_full_v2.py --minchar 50 --query pre2022 \
  --files "$F" --task 0 --ntask 1 --selfcheck --out $OUT/scan > $OUT/scan.log
apptainer exec $B $SIF python3 - $OUT/scan/summary_task0000.json <<'PY'
import json, sys
T = json.load(open(sys.argv[1]))
r = {k: T.get(k) for k in ('candidates_total', 'queries', 'query_accounts', 'index', 'setup_s', 'row_groups', 'rows', 'rows_date',
     'rows_verified', 'verified', 'pairs', 'queries_with_match', 'wall_s', 'cpu_s', 'max_rss_one_worker_GB', 'DONE')}
r['cpu_busy_of_4_cores'] = round(T['cpu_s'] / (4 * T['wall_s']), 2)
r['selfcheck_ok'] = all(x['ok'] for x in T.get('selfcheck', [])) and len(T.get('selfcheck', [])) > 0
print(json.dumps(r))
PY
echo "== scan end $(date +%T)"
# from here on a failing check does not stop the script; each prints its exit code (rc=)
set +e
echo "== regression (old queries vs A/B run r1_w4_A)"; apptainer exec $B $SIF python3 $D/compare_subset_v1.py $OLD $OUT/scan $L1; echo "rc=$?"
echo "== invariants"; apptainer exec $B $SIF python3 $D/invariants_check_v1.py $OUT/scan 50; echo "rc=$?"
echo "== brute force"; LSH_DIR=$L2 apptainer exec $B $SIF python3 $D/bruteforce_check_v1.py --out $OUT/scan --files "$F" --nrg-comment 2 --nrg-post 0 --nq 1000 --seed 1; echo "rc=$?"
echo "== done $(date +%T)"
