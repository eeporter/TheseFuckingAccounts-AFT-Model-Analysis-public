#!/usr/bin/env python3
"""compare_pairs_v1.py DIR_A DIR_B - are the pairs in two output folders identical as multisets (all 14 columns,
duplicates counted)? Also checks both summaries say DONE. Prints counts only.

Inputs: two output folders (summary_task*.json + pairs_task*_p*.parquet), here two runs of fast2_full_v2.py on the
same files. Output: one counts-only JSON on stdout. Exit 0 = identical, 3 = outputs differ, other = the check failed.
Job: run_ab_workers_v1.sh, 4-vs-12 worker A/B test 43332090 (run r1_w4_A vs r3_w12_A: identical pairs).

Method: every pair row becomes a Python tuple of all its columns, counted in a collections.Counter (a multiset).
Counter subtraction in both directions gives the rows only in A and only in B; both must be 0. Exact (no hashing);
fine for the small A/B outputs, too slow and memory-hungry for a full run (compare_full_v1.py hashes instead)."""
import sys, glob, json, collections
import pyarrow as pa, pyarrow.parquet as pq
out = {}
# per folder: summaries, all pair files concatenated, and a multiset of whole rows
for tag, d in (('A', sys.argv[1]), ('B', sys.argv[2])):
    s = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/summary_task*.json'))]
    fl = sorted(glob.glob(f'{d}/pairs_task*_p*.parquet'))
    t = pa.concat_tables([pq.read_table(f) for f in fl]) if fl else None
    rows = collections.Counter(zip(*[t[c].to_pylist() for c in t.column_names])) if t is not None else collections.Counter()
    out[tag] = dict(summaries=len(s), all_done=bool(s) and all(x.get('DONE') for x in s), pair_files=len(fl),
                    pairs=sum(rows.values()), rows_scanned=sum(x.get('rows', 0) for x in s), counter=rows,
                    columns=t.column_names if t is not None else [])
a, b = out['A'], out['B']
res = {'A_dir': sys.argv[1], 'B_dir': sys.argv[2],
       'A': {k: v for k, v in a.items() if k not in ('counter', 'columns')}, 'B': {k: v for k, v in b.items() if k not in ('counter', 'columns')},
       'same_columns': a['columns'] == b['columns'], 'only_in_A': sum((a['counter'] - b['counter']).values()),
       'only_in_B': sum((b['counter'] - a['counter']).values())}
# identical = both runs DONE, same columns, no row only on one side (with multiplicity), same archive rows read
res['IDENTICAL'] = bool(a['all_done'] and b['all_done'] and res['same_columns'] and res['only_in_A'] == 0 and res['only_in_B'] == 0
                        and a['rows_scanned'] == b['rows_scanned'])
print(json.dumps(res, indent=1))
sys.exit(0 if res['IDENTICAL'] else 3)   # 3 = outputs differ; any other non-zero = the comparison itself failed
