#!/usr/bin/env python3
"""fast2_full_v3.py (2026-11-18) = fast2_full_v2.py with one change, for row groups with very many pairs (mass-copied texts):
  rules3.scan_group_fast builds the hits in slices of --chunk pairs (no Arrow 2 GB string overflow), and a row group with
  more than --chunk pairs is written by the worker itself, slice by slice, as pairs_taskNNNN_pw<pid>_KKKKK.parquet
  (so it is never held in the parent). Matching, rules and output columns are unchanged.
  Also --only FILE:RG[,FILE:RG...] restricts the scan to those row groups (for testing); file assignment and index
  order still come from --task/--ntask, so pass --ntask 30 explicitly when rerunning single tasks of the 30-task run.
  The worker pool is a ProcessPoolExecutor: if a worker is killed (e.g. out of memory) the task stops with an error
  instead of hanging, and the summary lists row groups with more than --chunk pairs (file, row group, pairs).

fast2_full_v2.py (2026-11-18) = fast2_full_v1.py with two switches for the 4-vs-12 worker test, nothing else changed:
  POOL_WORKERS (env): number of worker processes (default SLURM_CPUS_PER_TASK, i.e. v1 behaviour);
  ARROW_CPU1=1 (env): each worker calls pa.set_cpu_count(1) so many workers do not each start a full Arrow thread pool.
It also records the peak memory of the largest worker (max_rss_one_worker_GB). The scan itself (rules2.scan_group_fast) is untouched.

Original v1 docstring: Exact copy search with all current rules, fastscan2 version, 2026-11-17. Same command line, same rules and same
output columns as fastscan's fast_full_v1.py (see rules2.py for the rules and for how they are applied with less work).
One Slurm array task processes every NTASK-th file of the selected archive files; row groups are spread over the
task's cores (fork; index built once).

Every earlier match is kept as its own row (one row per accused text x earlier archive item, with the archive item's
author, subreddit and time), so the number of earlier copies and of distinct earlier authors per text can be counted
afterwards. Output keeps both texts' visible length, so a run at --minchar 50 also yields the 140 set, and each query
text's own post time (q_created_utc) and accusation time (q_acc_utc).
Pairs are written as pairs_taskNNNN_pKKKK.parquet (a new part whenever --flush-rows pairs are held), plus
summary_taskNNNN.json with "DONE": true when the task finished.
Writes ids, metadata and scores only; prints counts only.

Inputs: LSH_DIR (default the project's lsh/ folder) with candidates_v1.parquet and sample_accounts_dates_v1.parquet
  (query side, see R.Queries); ARCH_DIR archive files (--files older = every older-batch comment and submission
  file, i.e. names without the NEWER uuids, sorted); command line --minchar, --query, --task/--ntask, --out.
Outputs (in --out): pairs_taskNNNN_pKKKK.parquet and summary_taskNNNN.json (counts, timings, index size, selfcheck).
Jobs: rerun of tasks 1, 3, 22 of the whole-pool run 43340006 = array 43360110 (run_fix_v3.sh: --ntask 30,
  --chunk 2000000, 12 workers, --selfcheck). Extra outputs: pairs_taskNNNN_pw<pid>_KKKKK.parquet written by workers.

Pipelines: the fast pipeline (fastscan2: C cleanup on raw Arrow buffers, early length skip, Bloom filter in front of
the inverted index; R.scan_group_fast) produces the output. The reference pipeline (fastscan's code in ref/,
R.scan_group_ref) is run only by --selfcheck, on row group 0 of the task's first comment file and first
submission file, and the task stops before scanning if the two pair sets differ."""
import argparse, glob, json, os, sys, time
import multiprocessing as mp
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastscan2 as fs
import rules3 as R

ap = argparse.ArgumentParser()
ap.add_argument('--minchar', type=int, default=140)
ap.add_argument('--query', choices=['pre2022', 'preacc', 'all'], default='pre2022')
ap.add_argument('--files', default='older', help="'older' = all older-batch files, or a comma list of paths")
ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
ap.add_argument('--ntask', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
ap.add_argument('--out', required=True)
ap.add_argument('--selfcheck', action='store_true', help='first row group of the first comment file and of the first post file of this task: also run fastscan\'s rule pipeline (ref/) and stop if the pairs differ')
ap.add_argument('--chunk', type=int, default=2000000, help='max pairs per hits slice; bigger row groups are written by the worker')
ap.add_argument('--only', default='', help='comma list of FILE:RG to scan (testing); default all row groups of this task')
ap.add_argument('--flush-rows', type=int, default=1000000, help='write pairs to disk whenever this many are held in memory')
args = ap.parse_args()

L = os.environ.get('LSH_DIR', '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh')
ARCH = os.environ.get('ARCH_DIR', '/gscratch/comdata/output')
NW = int(os.environ.get('POOL_WORKERS') or os.environ.get('SLURM_CPUS_PER_TASK', '4'))
ARROW_CPU1 = os.environ.get('ARROW_CPU1', '0') == '1'
os.makedirs(args.out, exist_ok=True)
T = {'args': vars(args), 'workers': NW, 'arrow_cpu1': ARROW_CPU1, 'cpus': os.environ.get('SLURM_CPUS_PER_TASK')}; t0 = time.time()

if args.files == 'older':
    files = sorted(f for d in ('reddit_comments_by_author.parquet', 'reddit_submissions_by_author.parquet')
                   for f in glob.glob(f'{ARCH}/{d}/*.parquet') if not any(u in f for u in R.NEWER))
else:
    files = args.files.split(',')
# file assignment: task k takes files k, k+ntask, k+2*ntask, ... of the sorted list (every NTASK-th file), so the
# same --task/--ntask always gives the same files (and therefore the same index order, see below)
mine = files[args.task::args.ntask]
T['files_total'] = len(files); T['files_this_task'] = len(mine)

Q = R.Queries(L, args.query, args.minchar)
T['candidates_total'] = Q.total; T['queries'] = Q.n; T['query_accounts'] = Q.accounts
# index order from an archive sample: shingles of row group 0 of this task's first file. The prefix filter indexes
# each query under its rarest shingles in this order. The prefix filter is exact under any fixed order, so this only
# changes speed. The index is built ONCE here, in the parent, before the worker pool is forked.
samp = R.sample_shingles(mine[0], 0, args.minchar) if mine else None    # orders the index; cannot change results
idx = fs.Index(fs.norm(Q.text), 7, 10, sample_shingles=samp)
del samp
T['index'] = {'query_shingles': idx.n_shingles, 'prefix_entries': idx.n_prefix}
T['setup_s'] = round(time.time() - t0, 1)


def init_worker():
    """Pool initializer: with ARROW_CPU1=1, limit each worker's Arrow thread pool to one thread (many workers would
    otherwise each start a full Arrow thread pool)."""
    if ARROW_CPU1:
        pa.set_cpu_count(1)


# per-process counter of worker-written pair files. Each forked worker has its own copy, so the file name also
# carries the worker's pid to keep names unique.
SINK = {'k': 0}


def sink(table):
    """Called inside a worker by rules3.scan_group_fast for each slice of a row group with more than --chunk pairs:
    writes the slice as pairs_taskNNNN_pw<pid>_KKKKK.parquet, so the big row group is never sent to the parent."""
    pq.write_table(table, f'{args.out}/pairs_task{args.task:04d}_pw{os.getpid()}_{SINK["k"]:05d}.parquet')
    SINK['k'] += 1


def work(job):
    """Worker: scan one row group job = (file path, row group number) with the fast pipeline.
    Returns (stats dict with cpu_s and wall_s added, hits table or None). Uses the module-level idx and Q, which the
    worker inherited from the parent through fork."""
    c0 = time.process_time(); w0 = time.time(); k0 = SINK['k']
    s, h = R.scan_group_fast(idx, Q, job[0], job[1], sink=sink, chunk=args.chunk)
    s['sink_parts'] = SINK['k'] - k0                   # pw files this row group wrote (0 for normal row groups)
    s['cpu_s'] = time.process_time() - c0; s['wall_s'] = time.time() - w0
    return s, h


def selfcheck():
    """fastscan's rule pipeline vs fastscan2 on one row group per kind. Returns a list of result dicts."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref'))
    import fastscan as fs1
    idx1 = fs1.Index(fs1.norm(Q.text), 7, 10)
    key = ['ckey', 'aid', 'akind', 'a_author', 'a_subreddit', 'a_created_utc', 'q_created_utc', 'q_acc_utc', 'q_vis_len', 'a_vis_len', 'inter', 'a_nsh', 'q_nsh']
    rows = lambda h: set() if h is None else set(zip(*[h[k].to_pylist() for k in key]))
    out = []
    for kind in ('comments', 'submissions'):
        f = next((x for x in mine if kind in x), None)
        if f is None: continue
        s1, h1 = R.scan_group_ref(fs1, idx1, Q, f, 0)
        s2, h2 = R.scan_group_fast(idx, Q, f, 0)
        a, b = rows(h1), rows(h2)
        out.append({'file': os.path.basename(f), 'rows': s1['rows'], 'rows_date': s1['rows_date'], 'ref_pairs': len(a), 'fast2_pairs': len(b),
                    'same': len(a & b), 'ok': bool(a == b and s1['rows_date'] == s2['rows_date'])})
    return out


if __name__ == '__main__':
    # self-check first: if fast and reference pipelines disagree, write the summary (without DONE) and exit
    if args.selfcheck:
        T['selfcheck'] = selfcheck()
        print(json.dumps({'selfcheck': T['selfcheck']}), flush=True)
        if not all(x['ok'] for x in T['selfcheck']):
            json.dump(T, open(f'{args.out}/summary_task{args.task:04d}.json', 'w'), indent=1, default=str)
            sys.exit('SELFCHECK FAILED: fastscan2 and fastscan rule pipeline disagree; nothing scanned')
    # one job per row group of every file of this task
    jobs = [(f, g) for f in mine for g in range(pq.ParquetFile(f).metadata.num_row_groups)]
    # --only (testing): keep just the listed FILE:RG row groups (full path or base name); file assignment and index
    # order are still those of --task/--ntask
    if args.only:
        want = {(a.rsplit(':', 1)[0], int(a.rsplit(':', 1)[1])) for a in args.only.split(',')}
        jobs = [j for j in jobs if (j[0], j[1]) in want or (os.path.basename(j[0]), j[1]) in want]
        assert len(jobs) == len(want), f'--only matched {len(jobs)} of {len(want)} row groups of this task'
    T['row_groups'] = len(jobs)
    t1 = time.time(); stats = []; parts = []; held = 0; nparts = 0; qhit = set()

    def flush():
        """Write the held pairs tables as the next pairs_taskNNNN_pKKKK.parquet part and reset the buffer.
        Called whenever --flush-rows pairs are held (bounds the parent's memory) and once at the end."""
        global parts, held, nparts
        if parts:
            pq.write_table(pa.concat_tables(parts), f'{args.out}/pairs_task{args.task:04d}_p{nparts:04d}.parquet')
            nparts += 1; parts = []; held = 0

    from concurrent.futures import ProcessPoolExecutor, as_completed    # v3: a killed worker raises BrokenProcessPool (task fails at once) instead of hanging
    # fork-based pool as in v2 (workers share Q and idx copy-on-write). With ProcessPoolExecutor, a worker killed
    # e.g. by the out-of-memory killer makes fut.result() raise BrokenProcessPool, so the task fails at once; the
    # multiprocessing.Pool of v1/v2 could wait forever for the lost result. Pairs returned to the parent are still
    # flushed every --flush-rows pairs; big row groups were already written by the worker (sink).
    with ProcessPoolExecutor(NW, mp_context=mp.get_context('fork'), initializer=init_worker) as pool:
        for fut in as_completed([pool.submit(work, j) for j in jobs]):
            s, h = fut.result()
            sq = s.pop('sunk_q', None)
            if sq is not None: qhit.update(Q.ckey[sq].astype(str).tolist())     # queries matched in row groups the worker wrote itself
            stats.append(s)
            if h is not None and h.num_rows:
                parts.append(h); held += h.num_rows; qhit.update(h['ckey'].to_pylist())
                if held >= args.flush_rows: flush()
    flush()
    T['pair_files'] = nparts; T['sink_pair_files'] = int(sum(s['sink_parts'] for s in stats))
    T['row_groups_sunk'] = int(sum(1 for s in stats if s['sink_parts']))
    # big_row_groups: [file, row group, pairs] of every row group with more than --chunk pairs, largest first
    T['big_row_groups'] = sorted(([s['file'], s['rg'], s['pairs']] for s in stats if s['pairs'] > args.chunk), key=lambda x: -x[2])   # counts only
    T['wall_s'] = round(time.time() - t1, 1)
    for k in ('rows', 'rows_date', 'rows_cleaned', 'rows_flagged', 'rows_verified', 'verified', 'pairs'):
        T[k] = int(sum(s[k] for s in stats))
    T['cpu_s'] = round(sum(s['cpu_s'] for s in stats), 1); T['worker_wall_s_sum'] = round(sum(s['wall_s'] for s in stats), 1)
    T['rows_per_s_per_core'] = round(T['rows'] / max(T['worker_wall_s_sum'], 1e-9))
    T['by_kind'] = {k: {x: int(sum(s[x] for s in stats if s['kind'] == k)) for x in ('rows', 'rows_date', 'pairs')} for k in ('comment', 'post')}
    T['queries_with_match'] = len(qhit)                 # distinct query texts (ckey) with at least one kept pair
    import resource
    T['max_rss_one_worker_GB'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)   # largest single worker (KiB on Linux); job total comes from the cgroup peak in the log
    T['total_s'] = round(time.time() - t0, 1)
    T['DONE'] = True                                    # marks a finished task; a summary without it means the task did not finish
    json.dump(T, open(f'{args.out}/summary_task{args.task:04d}.json', 'w'), indent=1, default=str)
    print(json.dumps(T, indent=1, default=str))
