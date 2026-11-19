#!/usr/bin/env python3
"""Equality + timing check of fastscan2 WITH ALL RULES against fastscan's rule pipeline, on the first N row groups of
one archive file. Both pipelines run in this one job on the same rows, single process, so the timings are comparable.
The reference side uses unmodified copies of fastscan.c / fastscan.py kept in ./ref (nothing in fastscan/ is touched).
Usage: rules_check_v1.py --file PATH --row-groups N [--minchar 140] [--query pre2022] [--lsh DIR] [--out DIR]
Prints counts and timings only. Writes nothing unless --out is given (then ids + scores, as fast_full_v1.py does).

Inputs: the archive file, the query folder (--lsh: candidates_v1.parquet + sample_accounts_dates_v1.parquet).
Reference side: rules2.scan_group_ref with ref/fastscan (date mask, +8 h, clean_and_keep, norm, prefix-filter scan,
earlier-only rule). Tested side: rules2.scan_group_fast with fastscan2 (C cleanup on raw buffers, Bloom filter, rows
containing "old reddit" or invalid UTF-8 sent through the old pipeline). Index order for fastscan2: shingle frequency
in row group 0 of --file (rarest first); this changes speed only.
Job: run_rules_check_v1.sh, job 43328587 (minchar 140 and 50).
PASS = the same set of pair rows (all KEY columns) and the same number of archive rows read on both sides."""
import argparse, json, os, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'ref'))
import fastscan as fs1
import fastscan2 as fs
import rules2 as R

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True); ap.add_argument('--row-groups', type=int, default=2)
ap.add_argument('--minchar', type=int, default=140); ap.add_argument('--query', default='pre2022')
ap.add_argument('--lsh', default='/gscratch/comdata/users/eeporter/reddit_copy_check/lsh'); ap.add_argument('--out', default='')
a = ap.parse_args()
T = {'file': os.path.basename(a.file), 'minchar': a.minchar, 'query': a.query}
Q = R.Queries(a.lsh, a.query, a.minchar)
T['candidates_total'] = Q.total; T['queries'] = Q.n; T['query_accounts'] = Q.accounts
nrg = pq.ParquetFile(a.file).metadata.num_row_groups
rgs = list(range(min(a.row_groups, nrg))); T['row_groups_in_file'] = nrg; T['row_groups_scanned'] = len(rgs)
qn = fs.norm(Q.text)
# every output column except jaccard (which is computed from inter, a_nsh, q_nsh)
KEY = ['ckey', 'aid', 'akind', 'a_author', 'a_subreddit', 'a_created_utc', 'q_created_utc', 'q_acc_utc', 'q_vis_len', 'a_vis_len', 'inter', 'a_nsh', 'q_nsh']


def rows_of(parts):
    """Set of KEY-column tuples of a list of hits tables (empty set if none). A set: duplicate rows would collapse."""
    if not parts: return set()
    t = pa.concat_tables(parts)
    return set(zip(*[t[k].to_pylist() for k in KEY]))


# reference: fastscan's pipeline
idx1 = fs1.Index(fs1.norm(Q.text), 7, 10)
c0 = time.process_time(); w0 = time.time(); ref = []; s1 = []
for rg in rgs:
    s, h = R.scan_group_ref(fs1, idx1, Q, a.file, rg); s1.append(s)
    if h is not None: ref.append(h)
T['ref_cpu_s'] = round(time.process_time() - c0, 2); T['ref_wall_s'] = round(time.time() - w0, 2)
for k in ('rows', 'rows_date', 'rows_len', 'verified', 'pairs'): T['ref_' + k] = int(sum(s[k] for s in s1))

# fastscan2
c0 = time.process_time()
idx = fs.Index(qn, 7, 10, sample_shingles=R.sample_shingles(a.file, 0, a.minchar))
T['fast2_sample_and_index_cpu_s'] = round(time.process_time() - c0, 2)
c0 = time.process_time(); w0 = time.time(); new = []; s2 = []
for rg in rgs:
    s, h = R.scan_group_fast(idx, Q, a.file, rg); s2.append(s)
    if h is not None: new.append(h)
T['fast2_cpu_s'] = round(time.process_time() - c0, 2); T['fast2_wall_s'] = round(time.time() - w0, 2)
for k in ('rows', 'rows_date', 'rows_cleaned', 'rows_flagged', 'rows_verified', 'verified', 'pairs'): T['fast2_' + k] = int(sum(s[k] for s in s2))

A, B = rows_of(ref), rows_of(new)
T['ref_pairs'] = len(A); T['fast2_pairs'] = len(B); T['same_pairs_all_columns'] = len(A & B)
T['ref_only'] = len(A - B); T['fast2_only'] = len(B - A)
T['speedup_cpu'] = round(T['ref_cpu_s'] / max(T['fast2_cpu_s'], 1e-9), 2)
T['ref_rows_per_s'] = round(T['ref_rows'] / max(T['ref_cpu_s'], 1e-9)); T['fast2_rows_per_s'] = round(T['fast2_rows'] / max(T['fast2_cpu_s'], 1e-9))
T['PASS'] = bool(A == B and sum(s['rows'] for s in s1) == sum(s['rows'] for s in s2))
if a.out:
    os.makedirs(a.out, exist_ok=True)
    if new: pq.write_table(pa.concat_tables(new), f'{a.out}/pairs_fast2.parquet')
    json.dump(T, open(f'{a.out}/summary.json', 'w'), indent=1, default=str)
print(json.dumps(T, indent=1, default=str))
