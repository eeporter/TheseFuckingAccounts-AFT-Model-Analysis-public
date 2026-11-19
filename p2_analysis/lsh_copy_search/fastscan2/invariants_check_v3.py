#!/usr/bin/env python3
"""invariants_check_v3.py OUT_DIR [MINCHAR] [TMPDIR] - invariants_check_v2.py plus two checks from the 18 Nov code review:
  self_match: a query text matched to its own archive row (ckey == akind:aid) must never appear (the earlier-only
    rule already excludes it; this checks that directly);
  rows_equal_summaries: the number of pair rows on disk must equal the sum of "pairs" in the summary_task*.json files.
Duplicates are still reported, not counted as a violation (see v2), but the count is printed for the README.

v2 docstring: same per-row rules as invariants_check_v1.py, but streaming, for
billions of rows (counts only). Reads one row group of one pair file at a time.
Duplicates (same ckey + akind:aid twice): each row's key is hashed to 64 bits and spilled to 256 bucket files on disk
(TMPDIR, deleted at the end); each bucket is then checked with np.unique. A 64-bit hash means at ~5B rows about one
false duplicate is possible by chance, so duplicates are reported (as in v1) but not counted as a violation.
Exit 0 = no violations, 3 = violations.

Inputs: OUT_DIR/pairs_task*_p*.parquet (all columns; the glob also matches worker-written
  pairs_task*_pw<pid>_*.parquet files) and OUT_DIR/summary_task*.json ("pairs" of each task); MINCHAR (default 50)
  for the visible-length test; TMPDIR (default OUT_DIR + '_inv_tmp') for the duplicate-hash bucket files.
Outputs: one JSON object on stdout (counts per check, duplicate_query_archive_pairs, pairs_in_summaries,
  VIOLATIONS). The bucket files in TMPDIR are temporary and removed at the end.
Jobs: checks_v3c 43361281 step 2 (on full_out_v3) and checks_v4 43362085 step 3 (on full_out_v4); both with
  MINCHAR 50.

Checks per row (each counted as a violation): jaccard_below_0p7 (10*inter < 7*(|Q|+|A|-inter)); jaccard_not_from_counts
(stored jaccard differs from inter/(|Q|+|A|-inter)); inter_gt_sizes (inter > |A|, inter > |Q| or inter < 1);
size_bound (length filter broken: 10|A| < 7|Q| or 7|A| > 10|Q|); archive_not_earlier (a_created_utc >= q_created_utc);
archive_after_cutoff / query_after_cutoff (time on/after 30 Nov 2022); q_vis_short / a_vis_short (visible length
< MINCHAR); nulls (any null value); self_match; plus rows_not_equal_summaries once for the whole folder. The
archive time check uses the stored a_created_utc, i.e. whatever time rule the run used."""
import sys, os, glob, json, shutil
import numpy as np, pandas as pd, pyarrow.parquet as pq
# same CUTOFF as rules2/rules3/rules4 (30 Nov 2022 00:00 UTC)
CUTOFF = int(np.datetime64('2022-11-30T00:00:00', 's').astype(np.int64))
OUT = sys.argv[1]
MINCHAR = int(sys.argv[2]) if len(sys.argv) > 2 else 50
TMP = sys.argv[3] if len(sys.argv) > 3 else OUT.rstrip('/') + '_inv_tmp'
# number of hash buckets (one temporary file each)
NB = 256
os.makedirs(TMP, exist_ok=True)
fh = [open(f'{TMP}/b{b:03d}.u64', 'wb') for b in range(NB)]
v = dict(files=0, rows=0, self_match=0, jaccard_below_0p7=0, jaccard_not_from_counts=0, inter_gt_sizes=0, size_bound=0,
         archive_not_earlier=0, archive_after_cutoff=0, query_after_cutoff=0, q_vis_short=0, a_vis_short=0, nulls=0)
for f in sorted(glob.glob(f'{OUT}/pairs_task*_p*.parquet')):
    pf = pq.ParquetFile(f); v['files'] += 1
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        v['nulls'] += sum(t[c].null_count for c in t.column_names)
        g = {c: t[c].to_numpy(zero_copy_only=False) for c in ('inter', 'a_nsh', 'q_nsh', 'jaccard', 'a_created_utc', 'q_created_utc', 'q_vis_len', 'a_vis_len')}
        i, a, q = g['inter'].astype(np.int64), g['a_nsh'].astype(np.int64), g['q_nsh'].astype(np.int64)
        v['rows'] += len(i)
        v['jaccard_below_0p7'] += int((10 * i < 7 * (q + a - i)).sum())
        v['jaccard_not_from_counts'] += int((np.abs(g['jaccard'] - i / (q + a - i)) > 1e-12).sum())
        v['inter_gt_sizes'] += int(((i > a) | (i > q) | (i < 1)).sum())
        # length filter in integers: a pair with Jaccard >= 0.7 must have 0.7|Q| <= |A| <= |Q|/0.7
        v['size_bound'] += int(((10 * a < 7 * q) | (7 * a > 10 * q)).sum())
        v['archive_not_earlier'] += int((g['a_created_utc'] >= g['q_created_utc']).sum())
        v['archive_after_cutoff'] += int((g['a_created_utc'] >= CUTOFF).sum())
        v['query_after_cutoff'] += int((g['q_created_utc'] >= CUTOFF).sum())
        v['q_vis_short'] += int((g['q_vis_len'] < MINCHAR).sum()); v['a_vis_short'] += int((g['a_vis_len'] < MINCHAR).sum())
        # self-match: the query key ckey is "kind:id", so a row whose archive item is the query text itself has ckey == akind:aid
        ak = t['akind'].to_pandas().astype(str) + ':' + t['aid'].to_pandas().astype(str)
        v['self_match'] += int((t['ckey'].to_pandas().astype(str).to_numpy() == ak.to_numpy()).sum())
        # duplicate check, step 1: hash each (query, archive item) key to 64 bits and append the hash to bucket file
        # h % NB. Equal keys give equal hashes and so always land in the same bucket.
        key = (t['ckey'].to_pandas().astype(str) + '|' + t['akind'].to_pandas().astype(str) + ':' + t['aid'].to_pandas().astype(str)).to_numpy()
        h = pd.util.hash_array(key.astype(object)).astype(np.uint64)
        # sort hashes by bucket; cuts[k]:cuts[k+1] is the slice for bucket k
        b = (h % NB).astype(np.int64); o = np.argsort(b, kind='stable'); hb = h[o]; cuts = np.searchsorted(b[o], np.arange(NB + 1))
        for k in range(NB):
            if cuts[k + 1] > cuts[k]: hb[cuts[k]:cuts[k + 1]].tofile(fh[k])
for x in fh: x.close()
# duplicate check, step 2: one bucket at a time fits in memory; duplicates in a bucket = hashes - distinct hashes.
# Summed over buckets this is the global number of repeated hashes (a few could be chance 64-bit collisions).
dup = 0
for k in range(NB):
    h = np.fromfile(f'{TMP}/b{k:03d}.u64', dtype=np.uint64)
    if len(h): dup += int(len(h) - len(np.unique(h)))
shutil.rmtree(TMP)
v['duplicate_query_archive_pairs'] = dup
v['pairs_in_summaries'] = int(sum(json.load(open(f)).get('pairs', 0) for f in glob.glob(f'{OUT}/summary_task*.json')))
v['rows_not_equal_summaries'] = int(v['rows'] != v['pairs_in_summaries'])
# every counter except the totals and the (report-only) duplicate count is a violation count
bad = sum(x for kk, x in v.items() if kk not in ('files', 'rows', 'duplicate_query_archive_pairs', 'pairs_in_summaries'))
v['VIOLATIONS'] = bad
print(json.dumps(v, indent=1))
sys.exit(0 if bad == 0 else 3)
