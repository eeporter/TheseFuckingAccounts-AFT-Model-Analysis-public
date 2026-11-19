#!/usr/bin/env python3
"""Pilot check of the exact copy search (fastscan), 2026-11-17: rerun the Spark pilot's COMMENT file with the pilot's rules
and check every Spark match comes back.
Inputs:
  lsh/candidates_v1.parquet               full copypasta candidate set (194,951 texts; pilot keeps those >= 140 characters)
  archive comment file = sorted(reddit_comments_by_author.parquet/*.parquet)[1]  (26.9M rows, same file as the pilot)
  lsh/pilot_out_v1/comment/pairs          Spark's scored pairs for that file
Pilot rules (lsh_pilot_v1.py): length(trim(text)) >= 140 on both sides, Spark cleanup, >= 3 words, word 3-grams,
dupe = exact Jaccard >= 0.7. No earlier-only / date / link / game-notice rules (those came later).
Parallel: row groups of the archive file split over N worker processes (fork; index built once in the parent).
Prints counts and timings only; writes ids + scores only (no text, no usernames).

Added notes (comment-only pass):
  Job: Hyak Slurm 43321861 (8 cores; PASS: all 427 Spark matches found with identical Jaccard, no extra pairs).
    Launched by run_pilot_fast_v1.sh.
  Outputs (pilot_fast_out_v1/ next to this file): pairs.parquet (all pairs found, with is_self flag),
    fast_only_pairs.parquet (non-self pairs found here but not among Spark's J >= 0.7 non-self pairs) and
    summary.json (counts, timings, PASS).
  PASS = every Spark non-self pair with J >= 0.7 is found AND Jaccard on the shared pairs differs by < 1e-9.
  Method: fastscan.Index (exact prefix filter, length filter, exact intersection); no Bloom filter in this version.
"""
import glob, json, os, sys, time
import multiprocessing as mp
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastscan as fs

L = os.environ.get('LSH_DIR', '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pilot_fast_out_v1'); os.makedirs(OUT, exist_ok=True)
COM = os.environ.get('COM_FILE') or sorted(glob.glob('/gscratch/comdata/output/reddit_comments_by_author.parquet/*.parquet'))[1]
NW = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
MINCHAR = 140
T = {'file': os.path.basename(COM), 'workers': NW}; t0 = time.time()


def long_enough(text):
    """Pilot length rule: boolean mask, length of the text with spaces trimmed (nulls as '') >= MINCHAR."""
    return pc.greater_equal(pc.utf8_length(pc.utf8_trim(pc.fill_null(text, ''), ' ')), MINCHAR)


# query side: full copypasta candidate set, pilot rule (>= 140 characters)
c = pq.read_table(f'{L}/candidates_v1.parquet', columns=['kind', 'id', 'text'])
T['candidates_total'] = c.num_rows
c = c.filter(long_enough(c['text']))
T['candidates_ge140'] = c.num_rows
ckey = np.array([f'{k}:{i}' for k, i in zip(c['kind'].to_pylist(), c['id'].to_pylist())], dtype=object)
cid = np.array(c['id'].to_pylist(), dtype=object); ckind = np.array(c['kind'].to_pylist(), dtype=object)
idx = fs.Index(fs.norm(c['text']), 7, 10)
del c
T['index'] = {'queries_with_shingles': int(idx.ok.sum()), 'query_shingles': idx.n_shingles,
              'prefix_entries': idx.n_prefix, 'distinct_prefix_keys': idx.n_keys}
T['index_build_s'] = round(time.time() - t0, 1)


def work(rg):
    """Scan one row group of the pilot comment file (in a forked worker).

    Applies the >= 140-character rule, normalises, scans against the index. Returns
    (rg, rows read, rows kept, rows shingled, pairs verified, seconds, archive ids, query ids, inter, m, jaccard).
    """
    t = time.time()
    tab = pq.ParquetFile(COM).read_row_group(rg, columns=['id', 'body'])
    n = tab.num_rows
    keep = long_enough(tab['body'])
    tab = tab.filter(keep)
    an = fs.norm(tab['body'])
    idx.stats[:] = 0
    r, q, inter, m, j = idx.scan(an)
    aid = np.array(tab['id'].to_pylist(), dtype=object)
    return (rg, n, tab.num_rows, int(idx.stats[0]), int(idx.stats[1]), time.time() - t,
            aid[r], q, inter, m, j)


if __name__ == '__main__':
    nrg = pq.ParquetFile(COM).metadata.num_row_groups
    T['row_groups'] = nrg
    t1 = time.time()
    with mp.get_context('fork').Pool(NW) as pool:
        res = pool.map(work, range(nrg), chunksize=1)
    T['scan_wall_s'] = round(time.time() - t1, 1)
    T['archive_rows'] = sum(x[1] for x in res); T['archive_rows_ge140'] = sum(x[2] for x in res)
    T['rows_shingled'] = sum(x[3] for x in res); T['pairs_verified'] = sum(x[4] for x in res)
    T['worker_cpu_s'] = round(sum(x[5] for x in res), 1)
    T['rows_per_s_per_core'] = round(T['archive_rows'] / T['worker_cpu_s'])
    aid = np.concatenate([x[6] for x in res]); q = np.concatenate([x[7] for x in res])
    inter = np.concatenate([x[8] for x in res]); m = np.concatenate([x[9] for x in res]); j = np.concatenate([x[10] for x in res])
    # self-match = the archive comment IS the query comment (same id, query kind comment); Spark flagged these too
    is_self = (cid[q] == aid) & (ckind[q] == 'comment')
    pq.write_table(pa.table({'ckey': ckey[q].astype(str), 'aid': aid.astype(str), 'inter': inter, 'a_nsh': m,
                             'q_nsh': idx.qlen[q], 'jaccard': j, 'is_self': is_self}), f'{OUT}/pairs.parquet')
    T['fast_pairs_j70'] = int(len(q)); T['fast_self'] = int(is_self.sum()); T['fast_non_self_j70'] = int((~is_self).sum())

    # compare with Spark's pilot pairs
    sp = pq.read_table(f'{L}/pilot_out_v1/comment/pairs', columns=['ckey', 'aid', 'jaccard', 'is_self']).to_pandas()
    s70 = sp[(sp.jaccard >= 0.7) & ~sp.is_self]
    sset = set(zip(s70.ckey, s70.aid))
    fj = {(k, a): v for k, a, v, s in zip(ckey[q], aid, j, is_self) if not s}
    spj = dict(zip(zip(sp.ckey, sp.aid), sp.jaccard))
    T['spark_non_self_j70'] = len(sset)
    T['spark_found_by_fast'] = len(sset & fj.keys())
    T['fast_only_non_self'] = len(fj.keys() - sset)
    # fast-only pairs that Spark nevertheless scored >= 0.7 in its pairs table
    T['fast_only_with_spark_score_ge70'] = len([k for k in fj.keys() - sset if spj.get(k, 0) >= 0.7])
    T['max_abs_jaccard_diff_shared'] = float(max([abs(fj[k] - spj[k]) for k in sset & fj.keys()] or [0.0]))
    T['spark_self_matches'] = int(sp.is_self.sum()); T['fast_self_matches'] = T['fast_self']
    only = sorted(fj.keys() - sset)
    pq.write_table(pa.table({'ckey': [k for k, _ in only], 'aid': [a for _, a in only], 'jaccard': [fj[k] for k in only]}),
                   f'{OUT}/fast_only_pairs.parquet')
    T['total_s'] = round(time.time() - t0, 1)
    T['PASS'] = bool(T['spark_found_by_fast'] == T['spark_non_self_j70'] and T['max_abs_jaccard_diff_shared'] < 1e-9)
    json.dump(T, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
    print(json.dumps(T, indent=1, default=str))
