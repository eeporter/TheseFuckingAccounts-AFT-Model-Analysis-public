#!/usr/bin/env python3
"""invariants_check_v1.py OUT_DIR - rules every output row must satisfy, checked on every row (counts only).
Exit 0 = no violations (duplicates are reported but not counted as a violation), 3 = violations.

Usage: invariants_check_v1.py OUT_DIR [MINCHAR (default 50)]. Reads every pairs_task*_p*.parquet in OUT_DIR whole.
Output: one counts-only JSON on stdout.
Jobs: check 4 of run_checks_v3.sh (job 43340007; on full_out_v3, ~5B rows, this version ran out of memory, hence
invariants_check_v2.py); also staged in run_scale_test_v1.sh (never submitted).

Rules checked on every pair row (each count must be 0):
  jaccard_below_0p7       Jaccard = c / (|A|+|B|-c) >= 0.7, tested in integers as 10c >= 7(|A|+|B|-c)
                          (c = inter, |A| = a_nsh, |B| = q_nsh shingle counts), so no rounding is involved.
  jaccard_not_from_counts the stored float jaccard equals inter / (q_nsh + a_nsh - inter).
  inter_gt_sizes          1 <= inter <= min(a_nsh, q_nsh).
  size_bound              length filter 0.7|B| <= |A| <= |B|/0.7, in integers 10a >= 7q and 7a <= 10q; any pair with
                          Jaccard >= 0.7 must satisfy it, so a violation would mean a wrong count.
  archive_not_earlier     earlier-only rule: a_created_utc < q_created_utc.
  archive_after_cutoff / query_after_cutoff   both times before 30 Nov 2022 (CUTOFF).
  q_vis_short / a_vis_short                    both visible lengths >= MINCHAR.
  nulls                   no null in any column.
Duplicates: the exact string key "ckey|akind:aid" of every row is kept in memory and counted with np.unique.
"""
import sys, glob, json
import numpy as np, pyarrow.parquet as pq
CUTOFF = int(np.datetime64('2022-11-30T00:00:00', 's').astype(np.int64))
MINCHAR = int(sys.argv[2]) if len(sys.argv) > 2 else 50
v = dict(rows=0, jaccard_below_0p7=0, jaccard_not_from_counts=0, inter_gt_sizes=0, size_bound=0,
         archive_not_earlier=0, archive_after_cutoff=0, query_after_cutoff=0, q_vis_short=0, a_vis_short=0, nulls=0)
keys = []                                   # exact duplicate keys of all rows (this is what grows with the output)
for f in sorted(glob.glob(f'{sys.argv[1]}/pairs_task*_p*.parquet')):
    t = pq.read_table(f)
    v['nulls'] += sum(t[c].null_count for c in t.column_names)
    g = {c: t[c].to_numpy(zero_copy_only=False) for c in ('inter', 'a_nsh', 'q_nsh', 'jaccard', 'a_created_utc', 'q_created_utc', 'q_vis_len', 'a_vis_len')}
    # int64 so 10 * i etc. cannot overflow
    i, a, q = g['inter'].astype(np.int64), g['a_nsh'].astype(np.int64), g['q_nsh'].astype(np.int64)
    v['rows'] += len(i)
    v['jaccard_below_0p7'] += int((10 * i < 7 * (q + a - i)).sum())
    v['jaccard_not_from_counts'] += int((np.abs(g['jaccard'] - i / (q + a - i)) > 1e-12).sum())
    v['inter_gt_sizes'] += int(((i > a) | (i > q) | (i < 1)).sum())
    v['size_bound'] += int(((10 * a < 7 * q) | (7 * a > 10 * q)).sum())
    v['archive_not_earlier'] += int((g['a_created_utc'] >= g['q_created_utc']).sum())
    v['archive_after_cutoff'] += int((g['a_created_utc'] >= CUTOFF).sum())
    v['query_after_cutoff'] += int((g['q_created_utc'] >= CUTOFF).sum())
    v['q_vis_short'] += int((g['q_vis_len'] < MINCHAR).sum()); v['a_vis_short'] += int((g['a_vis_len'] < MINCHAR).sum())
    keys.append(np.char.add(np.char.add(t['ckey'].to_numpy(zero_copy_only=False).astype(str), '|'),
                            np.char.add(np.char.add(t['akind'].to_numpy(zero_copy_only=False).astype(str), ':'), t['aid'].to_numpy(zero_copy_only=False).astype(str))))
k = np.concatenate(keys) if keys else np.array([], str)
v['duplicate_query_archive_pairs'] = int(len(k) - len(np.unique(k))) if len(k) else 0
# every rule count except the row and duplicate counts is a violation
bad = sum(x for kk, x in v.items() if kk not in ('rows', 'duplicate_query_archive_pairs'))
v['VIOLATIONS'] = bad
print(json.dumps(v, indent=1))
sys.exit(0 if bad == 0 else 3)
