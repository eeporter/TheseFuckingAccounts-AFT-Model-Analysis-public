#!/usr/bin/env python3
"""compare_subset_v1.py OLD_OUT NEW_OUT OLD_LSH_DIR - regression check for a run with a larger query set.
Every old query (rules2.Queries on OLD_LSH_DIR, pre2022, minchar 50) is also a new query, and the search is exact,
so the new output restricted to the old queries must equal the old output exactly (all 14 columns, as multisets).
Old queries that had no pairs must have none in the new output either. Prints counts only. Exit 0 = identical, 3 = not.

Inputs: OLD_OUT, NEW_OUT = full-run output folders (pairs_task*_p*.parquet); OLD_LSH_DIR = the old query folder
(lsh/, candidates_v1.parquet + sample_accounts_dates_v1.parquet) from which the old query keys are rebuilt.
Output: one counts-only JSON on stdout.
Jobs: check 2 ("regression", old 71,420 queries vs full_out_v2) of run_checks_v3.sh (job 43340007) and of
run_checks_v3b.sh (job 43360112); also staged in run_scale_test_v1.sh (never submitted).

Method: both outputs are filtered to rows whose ckey is an old query key, then compared as hash multisets: every row
(all columns) is hashed to 64 bits with pandas hash_pandas_object, the hashes are sorted, and the sorted arrays must be
equal. This is exact up to a 64-bit hash collision. Because the search is exact (prefix filter + exact Jaccard), adding
queries cannot add or remove pairs of the old queries; any difference would point to a bug or to different inputs."""
import sys, os, glob, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules2 as R
old, new, lsh_old = sys.argv[1:4]
# old query set, rebuilt with the same selection as the old run (pre2022 queries, minchar 50)
Q = R.Queries(lsh_old, 'pre2022', 50)
keyset = pa.array(sorted(set(Q.ckey.tolist())))
res = {'old_queries': int(Q.n)}
H = {}
for tag, d in (('old', old), ('new', new)):
    hs = []; cols = None; n_all = 0
    for f in sorted(glob.glob(f'{d}/pairs_task*_p*.parquet')):
        t = pq.read_table(f); n_all += t.num_rows; cols = cols or t.column_names
        # keep only pairs of old queries (n_all still counts every pair in the folder)
        t = t.filter(pc.is_in(t['ckey'], value_set=keyset))
        if t.num_rows:
            hs.append(pd.util.hash_pandas_object(t.select(cols).to_pandas(), index=False).to_numpy())
    H[tag] = np.sort(np.concatenate(hs)) if hs else np.empty(0, np.uint64)
    res[tag] = {'pairs_all': n_all, 'pairs_for_old_queries': int(len(H[tag])), 'columns': cols}
a, b = H['old'], H['new']
# equal sorted hash arrays = equal multisets of rows
same = len(a) == len(b) and bool(np.array_equal(a, b))
if not same:
    ua, ca = np.unique(a, return_counts=True); ub, cb = np.unique(b, return_counts=True)
    ia, ib = dict(zip(ua.tolist(), ca.tolist())), dict(zip(ub.tolist(), cb.tolist()))
    res['only_in_old'] = sum(max(0, c - ib.get(k, 0)) for k, c in ia.items())
    res['only_in_new'] = sum(max(0, c - ia.get(k, 0)) for k, c in ib.items())
res['IDENTICAL'] = bool(same and res['old']['columns'] == res['new']['columns'])
print(json.dumps(res, indent=1, default=str))
sys.exit(0 if res['IDENTICAL'] else 3)
