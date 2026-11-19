#!/usr/bin/env python3
"""Speed + equality check of fastscan2 against fastscan, 2026-11-17. Same file, same rules, same worker count as
fastscan's pilot (pilot_fast_v1.py, job 43321861), so the two timings can be compared directly.
Inputs:
  lsh/candidates_v1.parquet   full copypasta candidate set (194,951 texts; pilot rule keeps the 36,089 with >= 140 characters)
  archive comment file = sorted(reddit_comments_by_author.parquet/*.parquet)[1]   (26,922,660 rows, the Spark pilot's file)
  fastscan/pilot_fast_out_v1/pairs.parquet   fastscan's pairs for that file (read only; nothing in fastscan/ is changed)
Pilot rules: length(trim(text)) >= 140 on both sides, Spark cleanup, >= 3 words, word 3-grams, exact Jaccard >= 0.7.
No date / link / game-notice rules (this is a speed and equality check only).
MAX_ROW_GROUPS (env) limits the run to the first N row groups for a tiny test; the comparison is then restricted
to fastscan pairs whose archive id is in those row groups.
Prints counts and timings only; writes ids + scores only (no text, no usernames).

Outputs (in OUT_NAME, default fastscan2/pilot_fast2_out_v1/): pairs.parquet (ckey, aid, inter, a_nsh, q_nsh, jaccard,
is_self) and summary.json (the dict printed at the end, with PASS = same pairs and same counts as fastscan).
Jobs: run_pilot_fast2_v1.sh, tiny check 43324646 (MAX_ROW_GROUPS=2) and pilot file 43328586 (PASS, 427 pairs).
Env: LSH_DIR, REF_PAIRS, OUT_NAME, COM_FILE, SLURM_CPUS_PER_TASK (workers), MAX_ROW_GROUPS, ORDER.
Index: prefix filter with threshold 7/10 (each query indexes its first n - ceil(0.7 n) + 1 shingles in one global
order). ORDER=sample orders shingles by frequency in row group 0 of the archive file (rarest first); any fixed order
gives the same pairs, only the speed changes."""
import glob, json, os, sys, time
import multiprocessing as mp
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastscan2 as fs

ROOT = '/gscratch/comdata/users/eeporter/reddit_copy_check'
L = os.environ.get('LSH_DIR', f'{ROOT}/lsh')
REF = os.environ.get('REF_PAIRS', f'{ROOT}/fastscan/pilot_fast_out_v1/pairs.parquet')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('OUT_NAME', 'pilot_fast2_out_v1')); os.makedirs(OUT, exist_ok=True)
COM = os.environ.get('COM_FILE') or sorted(glob.glob('/gscratch/comdata/output/reddit_comments_by_author.parquet/*.parquet'))[1]
NW = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
MAXRG = int(os.environ.get('MAX_ROW_GROUPS', '0'))
ORDER = os.environ.get('ORDER', 'sample')          # 'sample' = index ordered by an archive sample; 'query' = fastscan's order
MINCHAR = 140
T = {'file': os.path.basename(COM), 'workers': NW, 'order': ORDER}; t0 = time.time()

c = pq.read_table(f'{L}/candidates_v1.parquet', columns=['kind', 'id', 'text'])
T['candidates_total'] = c.num_rows
# pilot length rule on the query side: >= 140 characters after trimming spaces
c = c.filter(pc.greater_equal(pc.utf8_length(pc.utf8_trim(pc.fill_null(c['text'], ''), ' ')), MINCHAR))
T['candidates_ge140'] = c.num_rows
ckey = np.array([f'{k}:{i}' for k, i in zip(c['kind'].to_pylist(), c['id'].to_pylist())], dtype=object)
cid = np.array(c['id'].to_pylist(), dtype=object); ckind = np.array(c['kind'].to_pylist(), dtype=object)
samp = None
if ORDER == 'sample':                               # first row group of the archive file as the sample
    ts = time.time()
    samp = fs.raw_shingles(pq.ParquetFile(COM).read_row_group(0, columns=['body'])['body'], MINCHAR)
    T['sample_rows'] = pq.ParquetFile(COM).metadata.row_group(0).num_rows; T['sample_s'] = round(time.time() - ts, 1)
# queries cleaned with the pyarrow cleanup (norm), then shingled, hashed and indexed (prefix filter, threshold 7/10)
idx = fs.Index(fs.norm(c['text']), 7, 10, sample_shingles=samp)
del c, samp
T['index'] = {'queries_with_shingles': int(idx.ok.sum()), 'query_shingles': idx.n_shingles,
              'prefix_entries': idx.n_prefix, 'distinct_prefix_keys': idx.n_keys}
T['index_build_s'] = round(time.time() - t0, 1)


def work(rg):
    """Scan one archive row group (runs in a forked worker, sharing the global index copy-on-write).
    Returns (row group, rows, idx.stats counters, wall s, cpu s, archive ids, query positions, inter, a_nsh, jaccard).
    scan_raw cleans the raw body buffer in C, skips rows under MINCHAR, Bloom-filters shingles before the inverted index
    lookup, and verifies candidates by exact Jaccard >= 0.7."""
    t = time.time(); c0 = time.process_time()
    f = pq.ParquetFile(COM)
    body = f.read_row_group(rg, columns=['body'])['body']
    # per-call counters (rows cleaned, rows verified, pairs verified, invalid UTF-8 rows), see T[...] below
    idx.stats[:] = 0
    r, q, inter, m, j = idx.scan_raw(body, minchar=MINCHAR)
    aid = np.empty(0, dtype=object)
    if len(r):                                       # ids are read only for row groups that have matches
        aid = np.array(f.read_row_group(rg, columns=['id'])['id'].take(pa.array(r)).to_pylist(), dtype=object)
    return (rg, len(body), [int(x) for x in idx.stats], time.time() - t, time.process_time() - c0, aid, q, inter, m, j)


if __name__ == '__main__':
    nrg = pq.ParquetFile(COM).metadata.num_row_groups
    T['row_groups_in_file'] = nrg
    rgs = list(range(nrg if not MAXRG else min(MAXRG, nrg))); T['row_groups_scanned'] = len(rgs)
    t1 = time.time()
    with mp.get_context('fork').Pool(NW) as pool:
        res = pool.map(work, rgs, chunksize=1)
    T['scan_wall_s'] = round(time.time() - t1, 1)
    T['archive_rows'] = sum(x[1] for x in res)
    T['rows_ge140_cleaned'] = sum(x[2][0] for x in res); T['rows_sorted_and_verified'] = sum(x[2][1] for x in res)
    T['pairs_verified'] = sum(x[2][2] for x in res); T['invalid_utf8_rows'] = sum(x[2][3] for x in res)
    T['worker_wall_s_sum'] = round(sum(x[3] for x in res), 1); T['worker_cpu_s'] = round(sum(x[4] for x in res), 1)
    T['rows_per_s_per_core'] = round(T['archive_rows'] / max(1e-9, T['worker_wall_s_sum']))
    aid = np.concatenate([x[5] for x in res]); q = np.concatenate([x[6] for x in res])
    inter = np.concatenate([x[7] for x in res]); m = np.concatenate([x[8] for x in res]); j = np.concatenate([x[9] for x in res])
    # self-match = the query comment itself found in the archive (no earlier-only rule in the pilot)
    is_self = (cid[q] == aid) & (ckind[q] == 'comment') if len(q) else np.zeros(0, bool)
    pq.write_table(pa.table({'ckey': ckey[q].astype(str), 'aid': aid.astype(str), 'inter': inter, 'a_nsh': m,
                             'q_nsh': idx.qlen[q], 'jaccard': j, 'is_self': is_self}), f'{OUT}/pairs.parquet')
    T['fast2_pairs_j70'] = int(len(q)); T['fast2_non_self_j70'] = int((~is_self).sum())

    # compare with fastscan's pairs for the same file (all pairs, self included)
    ref = pq.read_table(REF, columns=['ckey', 'aid', 'inter', 'a_nsh', 'q_nsh']).to_pandas()
    if MAXRG:                                        # restrict fastscan's pairs to the row groups scanned here
        ids = set()
        for rg in rgs: ids.update(pq.ParquetFile(COM).read_row_group(rg, columns=['id'])['id'].to_pylist())
        ref = ref[ref.aid.isin(ids)]
    # compared as sets of (ckey, aid, inter, a_nsh, q_nsh): same pairs AND same shingle counts
    a = set(zip(ref.ckey, ref.aid, ref.inter, ref.a_nsh, ref.q_nsh))
    b = set(zip(ckey[q].astype(str).tolist(), aid.astype(str).tolist(), inter.tolist(), m.tolist(), idx.qlen[q].tolist()))
    T['fastscan_pairs_in_scope'] = len(a); T['same_pairs_and_counts'] = len(a & b)
    T['fastscan_only'] = len(a - b); T['fast2_only'] = len(b - a)
    T['total_s'] = round(time.time() - t0, 1)
    T['PASS'] = bool(a == b)
    json.dump(T, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
    print(json.dumps(T, indent=1, default=str))
