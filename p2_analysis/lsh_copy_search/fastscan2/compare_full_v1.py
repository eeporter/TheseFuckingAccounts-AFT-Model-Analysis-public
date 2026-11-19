#!/usr/bin/env python3
"""compare_full_v1.py DIR_A DIR_B - are two full-run output folders identical? Every pair row (all 14 columns) is
hashed to one 64-bit number; the sorted hash lists are compared, so duplicates count and the task layout does not
matter. Also checks every summary says DONE, the task file counts add up to 2,000 and the rows scanned match.
Prints counts only.

Inputs: two output folders of the full copy search (summary_task*.json + pairs_task*_p*.parquet). The glob also matches
the worker-written pairs_taskNNNN_pw<pid>_KKKKK.parquet files of fast2_full_v3.py.
Output: one counts-only JSON on stdout. Exit 0 = IDENTICAL, 3 = not identical (or incomplete).
Job: run_compare_full_v1.sh, job 43332603 (full_out_v1 from array 43329112 vs full_out_v2 from array 43332591):
IDENTICAL, 24,277,320 pairs.

Method (hash-multiset comparison): each pair row is reduced to one 64-bit hash of all its columns
(pandas hash_pandas_object, index ignored). Sorting the hashes of each side and comparing the two sorted arrays is the
same as comparing the two multisets of rows, without holding 24M Python tuples in memory. Row order and which task or
file a row is in do not matter; a row present twice must be present twice on the other side too. The only way two
different outputs could look equal is a 64-bit hash collision that exactly cancels out, which is negligible."""
import sys, glob, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
res = {}
H = {}
# one pass per folder: read the summaries, hash every pair row, keep the sorted hashes
for tag, d in (('A', sys.argv[1]), ('B', sys.argv[2])):
    s = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/summary_task*.json'))]
    hs = []; cols = None
    for f in sorted(glob.glob(f'{d}/pairs_task*_p*.parquet')):
        t = pq.read_table(f); cols = cols or t.column_names
        # t.select(cols): hash the columns in the first file's order, so a different column order cannot change hashes
        hs.append(pd.util.hash_pandas_object(t.select(cols).to_pandas(), index=False).to_numpy())
    h = np.sort(np.concatenate(hs)) if hs else np.empty(0, np.uint64)
    H[tag] = h
    res[tag] = {'summaries': len(s), 'all_done': bool(s) and all(x.get('DONE') for x in s),
                'files_scanned': sum(x.get('files_this_task', 0) for x in s), 'rows_scanned': sum(x.get('rows', 0) for x in s),
                'pairs_in_summaries': sum(x.get('pairs', 0) for x in s), 'pair_rows_on_disk': int(len(h)),
                'max_task_wall_s': max((x.get('total_s', 0) for x in s), default=0), 'columns': cols}
a, b = H['A'], H['B']
# equal sorted hash arrays = equal multisets of rows
same = len(a) == len(b) and bool(np.array_equal(a, b))
if not same:
    # count how many rows (with multiplicity) each side has beyond the other
    ua, ca = np.unique(a, return_counts=True); ub, cb = np.unique(b, return_counts=True)
    ia = dict(zip(ua.tolist(), ca.tolist())); ib = dict(zip(ub.tolist(), cb.tolist()))
    res['only_in_A'] = sum(max(0, c - ib.get(k, 0)) for k, c in ia.items())
    res['only_in_B'] = sum(max(0, c - ia.get(k, 0)) for k, c in ib.items())
ra, rb = res['A'], res['B']
# identical only if the pairs match AND both runs are complete (every task DONE, all 2,000 archive files scanned,
# same number of archive rows read) AND the column lists match
res['IDENTICAL'] = bool(same and ra['all_done'] and rb['all_done'] and ra['files_scanned'] == rb['files_scanned'] == 2000
                        and ra['rows_scanned'] == rb['rows_scanned'] and ra['columns'] == rb['columns'])
print(json.dumps(res, indent=1, default=str))
sys.exit(0 if res['IDENTICAL'] else 3)
