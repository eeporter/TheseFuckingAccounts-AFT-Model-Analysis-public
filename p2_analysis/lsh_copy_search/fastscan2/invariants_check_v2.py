#!/usr/bin/env python3
"""invariants_check_v2.py OUT_DIR [MINCHAR] [TMPDIR] - same per-row rules as invariants_check_v1.py, but streaming, for
billions of rows (counts only). Reads one row group of one pair file at a time.
Duplicates (same ckey + akind:aid twice): each row's key is hashed to 64 bits and spilled to 256 bucket files on disk
(TMPDIR, deleted at the end); each bucket is then checked with np.unique. A 64-bit hash means at ~5B rows about one
false duplicate is possible by chance, so duplicates are reported (as in v1) but not counted as a violation.
Exit 0 = no violations, 3 = violations.

Inputs: OUT_DIR with pairs_task*_p*.parquet (the glob also matches fast2_full_v3.py's worker-written
pairs_taskNNNN_pw<pid>_KKKKK.parquet); MINCHAR default 50; TMPDIR default OUT_DIR + '_inv_tmp'.
Output: one counts-only JSON on stdout. Job: check 4 of run_checks_v3b.sh, job 43360112 (merged full_out_v3).

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

Streaming: memory is bounded by one row group of one pair file plus 256 open file handles. The per-row rule counts are
plain sums, so they need no state; only the duplicate check needs all keys, and it is done on disk in two passes:
pass 1 appends each row's 64-bit key hash to bucket file (hash mod 256); pass 2 loads one bucket at a time. Two equal
keys have equal hashes and so land in the same bucket, so counting duplicates bucket by bucket finds every duplicate.
"""
import sys, os, glob, json, shutil
import numpy as np, pandas as pd, pyarrow.parquet as pq
CUTOFF = int(np.datetime64('2022-11-30T00:00:00', 's').astype(np.int64))
OUT = sys.argv[1]
MINCHAR = int(sys.argv[2]) if len(sys.argv) > 2 else 50
TMP = sys.argv[3] if len(sys.argv) > 3 else OUT.rstrip('/') + '_inv_tmp'
NB = 256
os.makedirs(TMP, exist_ok=True)
# one append-only binary file per bucket, raw uint64 hashes
fh = [open(f'{TMP}/b{b:03d}.u64', 'wb') for b in range(NB)]
v = dict(files=0, rows=0, jaccard_below_0p7=0, jaccard_not_from_counts=0, inter_gt_sizes=0, size_bound=0,
         archive_not_earlier=0, archive_after_cutoff=0, query_after_cutoff=0, q_vis_short=0, a_vis_short=0, nulls=0)
for f in sorted(glob.glob(f'{OUT}/pairs_task*_p*.parquet')):
    pf = pq.ParquetFile(f); v['files'] += 1
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
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
        # duplicate key "ckey|akind:aid" (same as v1), hashed to 64 bits
        key = (t['ckey'].to_pandas().astype(str) + '|' + t['akind'].to_pandas().astype(str) + ':' + t['aid'].to_pandas().astype(str)).to_numpy()
        h = pd.util.hash_array(key.astype(object)).astype(np.uint64)
        # group the hashes by bucket (stable sort by bucket number, then cut points) and append each group to its file
        b = (h % NB).astype(np.int64); o = np.argsort(b, kind='stable'); hb = h[o]; cuts = np.searchsorted(b[o], np.arange(NB + 1))
        for k in range(NB):
            if cuts[k + 1] > cuts[k]: hb[cuts[k]:cuts[k + 1]].tofile(fh[k])
for x in fh: x.close()
# pass 2: duplicates inside each bucket = rows minus distinct hashes
dup = 0
for k in range(NB):
    h = np.fromfile(f'{TMP}/b{k:03d}.u64', dtype=np.uint64)
    if len(h): dup += int(len(h) - len(np.unique(h)))
shutil.rmtree(TMP)                          # bucket files deleted at the end
v['duplicate_query_archive_pairs'] = dup
# every rule count except files, rows and duplicates is a violation
bad = sum(x for kk, x in v.items() if kk not in ('files', 'rows', 'duplicate_query_archive_pairs'))
v['VIOLATIONS'] = bad
print(json.dumps(v, indent=1))
sys.exit(0 if bad == 0 else 3)
