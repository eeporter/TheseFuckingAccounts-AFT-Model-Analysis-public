#!/usr/bin/env python3
"""Exact copy search with all current rules, fastscan2 version, 2026-11-17. Same command line, same rules and same
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
Jobs: first full run, Slurm array 43329112 (run_full_fast2_v1.sh: 987-account query set, 30 tasks, --minchar 50,
  --query pre2022, --selfcheck).

Pipelines: the fast pipeline (fastscan2: C cleanup on raw Arrow buffers, early length skip, Bloom filter in front of
the inverted index; R.scan_group_fast) produces the output. The reference pipeline (fastscan's code in ref/,
R.scan_group_ref) is run only by --selfcheck, on row group 0 of the task's first comment file and first
submission file, and the task stops before scanning if the two pair sets differ."""
import argparse, glob, json, os, sys, time
import multiprocessing as mp
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastscan2 as fs
import rules2 as R

ap = argparse.ArgumentParser()
ap.add_argument('--minchar', type=int, default=140)
ap.add_argument('--query', choices=['pre2022', 'preacc', 'all'], default='pre2022')
ap.add_argument('--files', default='older', help="'older' = all older-batch files, or a comma list of paths")
ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
ap.add_argument('--ntask', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
ap.add_argument('--out', required=True)
ap.add_argument('--selfcheck', action='store_true', help='first row group of the first comment file and of the first post file of this task: also run fastscan\'s rule pipeline (ref/) and stop if the pairs differ')
ap.add_argument('--flush-rows', type=int, default=1000000, help='write pairs to disk whenever this many are held in memory')
args = ap.parse_args()

L = os.environ.get('LSH_DIR', '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh')
ARCH = os.environ.get('ARCH_DIR', '/gscratch/comdata/output')
NW = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
os.makedirs(args.out, exist_ok=True)
T = {'args': vars(args), 'workers': NW}; t0 = time.time()

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


def work(job):
    """Worker: scan one row group job = (file path, row group number) with the fast pipeline.
    Returns (stats dict with cpu_s and wall_s added, hits table or None). Uses the module-level idx and Q, which the
    worker inherited from the parent through fork."""
    c0 = time.process_time(); w0 = time.time()
    s, h = R.scan_group_fast(idx, Q, job[0], job[1])
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
    T['row_groups'] = len(jobs)
    t1 = time.time(); stats = []; parts = []; held = 0; nparts = 0; qhit = set()

    def flush():
        """Write the held pairs tables as the next pairs_taskNNNN_pKKKK.parquet part and reset the buffer.
        Called whenever --flush-rows pairs are held (bounds the parent's memory) and once at the end."""
        global parts, held, nparts
        if parts:
            pq.write_table(pa.concat_tables(parts), f'{args.out}/pairs_task{args.task:04d}_p{nparts:04d}.parquet')
            nparts += 1; parts = []; held = 0

    # fork-based pool: workers are forked after Q and idx (including the C-side index set up by fs.Index) were built,
    # so they share them copy-on-write instead of rebuilding or pickling them. Results come back in any order; pairs
    # are collected in the parent and flushed to disk every --flush-rows pairs.
    with mp.get_context('fork').Pool(NW) as pool:
        for s, h in pool.imap_unordered(work, jobs, chunksize=1):
            stats.append(s)
            if h is not None and h.num_rows:
                parts.append(h); held += h.num_rows; qhit.update(h['ckey'].to_pylist())
                if held >= args.flush_rows: flush()
    flush()
    T['pair_files'] = nparts
    T['wall_s'] = round(time.time() - t1, 1)
    for k in ('rows', 'rows_date', 'rows_cleaned', 'rows_flagged', 'rows_verified', 'verified', 'pairs'):
        T[k] = int(sum(s[k] for s in stats))
    T['cpu_s'] = round(sum(s['cpu_s'] for s in stats), 1); T['worker_wall_s_sum'] = round(sum(s['wall_s'] for s in stats), 1)
    T['rows_per_s_per_core'] = round(T['rows'] / max(T['worker_wall_s_sum'], 1e-9))
    T['by_kind'] = {k: {x: int(sum(s[x] for s in stats if s['kind'] == k)) for x in ('rows', 'rows_date', 'pairs')} for k in ('comment', 'post')}
    T['queries_with_match'] = len(qhit)                 # distinct query texts (ckey) with at least one kept pair
    T['total_s'] = round(time.time() - t0, 1)
    T['DONE'] = True                                    # marks a finished task; a summary without it means the task did not finish
    json.dump(T, open(f'{args.out}/summary_task{args.task:04d}.json', 'w'), indent=1, default=str)
    print(json.dumps(T, indent=1, default=str))
