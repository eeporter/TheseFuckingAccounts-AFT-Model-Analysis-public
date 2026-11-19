#!/usr/bin/env bash
# 4-vs-12 worker A/B test: the real copy search (fast2_full_v2.py, all rules, minchar 50, pre2022 queries, self-check on).
# Older-batch comment files at sorted positions 620-623 (io_probe_v1 uses 600-616; earlier tests used other files).
#   run 1: 4 workers, exactly v1, on 620+621  (first read on this node -> timing)
#   run 2: 12 workers + 1 Arrow thread each, on 622+623 (first read on this node -> timing)
#   run 3: 12 workers on 620+621 again, only to check its pairs equal run 1 (cache cannot change results)
# Timing is compared per row and per GB read, because the two file pairs differ in size.
# Pair files stay in fastscan2/ab_workers_v1/ on Hyak; the log has counts only.
# Job: 43332090 (pairs of 4 and 12 workers identical). Uses fast2_full_v2.py and compare_pairs_v1.py.
# Inputs: older-batch comment files (uuid 074ac9d9) at sorted positions 620-623, query set lsh/ (fast2_full_v2.py default).
# Outputs: ab_workers_v1/<job id>/<run>/ (pairs + summary), <run>.log, and one counts-only JSON line per run in the job log.
# Submit: sbatch -A comdata -p cpu-g2 -c 4 --mem 32G --time 0:20:00 -J ab_workers_v1 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/ab_workers_v1_%j.out run_ab_workers_v1.sh
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
OUT=$D/ab_workers_v1/${SLURM_JOB_ID:-x}
# Private build folder per job/task: the C code is compiled here, outside the container, and the Python modules are
# pointed at it through FASTSCAN2_LIB / FASTSCAN_LIB. Concurrent jobs or array tasks must not share one .so that
# another task may be rewriting while it is loaded.
BLD=$D/build/ab_${SLURM_JOB_ID:-x}
mkdir -p $BLD $OUT
cd $D
# the 1,200 older-batch comment files in byte order (LC_ALL=C), so positions 620-623 are well defined
mapfile -t ALL < <(ls /gscratch/comdata/output/reddit_comments_by_author.parquet/*074ac9d9*.parquet | LC_ALL=C sort)
[ ${#ALL[@]} -eq 1200 ] || { echo "expected 1200 older comment files, found ${#ALL[@]}"; exit 1; }
FA="${ALL[620]},${ALL[621]}"; FB="${ALL[622]},${ALL[623]}"
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-x} using comment file positions 620-623"
# libfastscan2.so = fastscan2 scanner; libfastscan.so = the unchanged fastscan (ref/) used by --selfcheck.
gcc -O3 -shared -fPIC -o $BLD/libfastscan2.so $D/fastscan2.c
gcc -O3 -shared -fPIC -o $BLD/libfastscan.so $D/ref/fastscan.c
export FASTSCAN2_LIB=$BLD/libfastscan2.so FASTSCAN_LIB=$BLD/libfastscan.so
# run NAME WORKERS ARROW_CPU1 FILES: one task (--task 0 --ntask 1) of fast2_full_v2.py over FILES into $OUT/NAME,
# POOL_WORKERS worker processes, ARROW_CPU1=1 = one Arrow thread per worker; then the Python below prints the
# summary plus GB of CreatedAt+body column chunks read (compressed), MB/s, rows/s, and whether the self-check passed.
run() {   # name workers arrow_cpu1 files
  echo "== $1: workers $2 arrow_cpu1=$3 start $(date +%T)"
  POOL_WORKERS=$2 ARROW_CPU1=$3 apptainer exec $B $SIF python3 $D/fast2_full_v2.py --minchar 50 --query pre2022 \
    --files "$4" --task 0 --ntask 1 --selfcheck --out $OUT/$1 > $OUT/$1.log
  apptainer exec $B $SIF python3 - $OUT/$1/summary_task0000.json "$4" <<'PY'
import json, sys, pyarrow.parquet as pq
T = json.load(open(sys.argv[1]))
gb = 0
for f in sys.argv[2].split(','):                        # compressed bytes of the columns the scan reads
    md = pq.ParquetFile(f).metadata
    names = [md.schema.column(i).path for i in range(md.num_columns)]
    for g in range(md.num_row_groups):
        gb += sum(md.row_group(g).column(names.index(c)).total_compressed_size for c in ('CreatedAt', 'body')) / 1e9
r = {k: T.get(k) for k in ('workers', 'arrow_cpu1', 'row_groups', 'rows', 'rows_date', 'pairs', 'queries_with_match',
     'setup_s', 'wall_s', 'cpu_s', 'max_rss_one_worker_GB', 'DONE')}
r['GB_read'] = round(gb, 2); r['MB_per_s'] = round(gb * 1e3 / T['wall_s'], 1); r['Mrows_per_s'] = round(T['rows'] / T['wall_s'] / 1e6, 3)
r['cpu_busy_of_4_cores'] = round(T['cpu_s'] / (4 * T['wall_s']), 2)
r['selfcheck_ok'] = all(x['ok'] for x in T.get('selfcheck', [])) and len(T.get('selfcheck', [])) > 0
print(json.dumps(r))
PY
  echo "== $1 end $(date +%T)"
}
# r1 and r3 read the same files (620, 621), so their pairs must be identical; r2 gives a cold-read timing at 12 workers
run r1_w4_A 4 0 "$FA"
run r2_w12_B 12 1 "$FB"
run r3_w12_A 12 1 "$FA"
# compare_pairs_v1.py exits 3 if the outputs differ; set +e so that code can be reported instead of aborting
set +e
apptainer exec $B $SIF python3 $D/compare_pairs_v1.py $OUT/r1_w4_A $OUT/r3_w12_A; rc=$?
set -e
case $rc in 0) echo "== PAIRS IDENTICAL (4 vs 12 workers)";; 3) echo "== OUTPUTS DIFFER"; exit 3;; *) echo "== COMPARISON FAILED rc=$rc"; exit $rc;; esac
echo "== done $(date +%T)"
