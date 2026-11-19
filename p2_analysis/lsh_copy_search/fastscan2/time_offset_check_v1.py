#!/usr/bin/env python3
"""time_offset_check_v1.py - is the archive's CreatedAt always UTC minus 8 h, or Pacific wall-clock time with daylight
saving (UTC minus 7 h from March to November)? Counts only; reads no text.

The copy search turns archive CreatedAt into UTC by adding a fixed 8 h. This checks that against an independent clock:
the query texts (candidates_v1.parquet in LSH_DIR) carry Reddit's own created_utc, and many of them are also rows of
the archive (same kind and id). For every query text found in the sampled archive files it computes
    offset = created_utc - CreatedAt (seconds)
and tabulates the offsets by calendar month (of created_utc). If CreatedAt is fixed UTC-8, every offset is 28,800 s in
every month. If it is US Pacific time with daylight saving, months in daylight time show 25,200 s (7 h).

Usage: time_offset_check_v1.py [--ncomment 40] [--npost 20] [--seed 1]   (LSH_DIR, ARCH_DIR as in the search)
Reads only the id and CreatedAt columns of --ncomment comment files and --npost post files of the older batch.
Exit 0 always; the verdict is in the printed JSON (offsets_seconds_by_month, share_8h, share_7h, verdict).

Inputs: LSH_DIR/candidates_v1.parquet (columns kind, id, created_utc only); ARCH_DIR older-batch comment and
  submission files (columns id, CreatedAt only), --ncomment / --npost of them drawn at random (--seed).
Outputs: one JSON object on stdout: query_texts, files per kind, matched_rows, offsets_seconds_all (10 most common
  offsets), offsets_seconds_by_month (4 most common per month), share_8h, share_7h, verdict, seconds. Nothing on disk.
Jobs: checks_v3c 43361281 step 1 (run_checks_v3c.sh: --ncomment 40 --npost 20 --seed 1, LSH_DIR = lsh_v2).
  Result recorded in rules4.py: 227,893 matched rows, offsets 8 h (61%) or 7 h (39%), split at the US clock changes,
  i.e. CreatedAt is America/Los_Angeles local time with daylight saving.

How offsets are computed: R.to_epoch_s on the naive CreatedAt timestamp reads its wall-clock digits as if they were
UTC, so created_utc - that value = (UTC - Pacific local time) = 28,800 s in standard time, 25,200 s in daylight time.
Each offset is counted under the calendar month (UTC) of created_utc; other offsets (if any) are counted as they are."""
import argparse, glob, json, os, sys, collections, time
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rules2 as R                                     # only for to_epoch_s and the batch uuids

ap = argparse.ArgumentParser()
ap.add_argument('--ncomment', type=int, default=40)
ap.add_argument('--npost', type=int, default=20)
ap.add_argument('--seed', type=int, default=1)
args = ap.parse_args()
L = os.environ.get('LSH_DIR', '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh')
ARCH = os.environ.get('ARCH_DIR', '/gscratch/comdata/output')
t0 = time.time(); rng = np.random.default_rng(args.seed)

# query side: kind, id and Reddit's own created_utc (true UTC), keyed by (kind, id)
c = pq.read_table(f'{L}/candidates_v1.parquet', columns=['kind', 'id', 'created_utc'])
q_utc = R.to_epoch_s(c['created_utc'])
# qmap[kind][id] = created_utc (if a (kind, id) occurs twice in candidates, the last one wins)
qmap = {k: dict() for k in ('comment', 'post')}
for k, i, u in zip(c['kind'].to_pylist(), c['id'].to_pylist(), q_utc.tolist()):
    qmap.setdefault(k, {})[i] = u

res = {'query_texts': c.num_rows, 'files': {}}
by_month = collections.defaultdict(collections.Counter)   # 'YYYY-MM' -> Counter(offset seconds)
for kind, d, n in (('comment', 'reddit_comments_by_author.parquet', args.ncomment), ('post', 'reddit_submissions_by_author.parquet', args.npost)):
    files = sorted(f for f in glob.glob(f'{ARCH}/{d}/*.parquet') if not any(u in f for u in R.NEWER))
    # n distinct random files of this kind (sorted list, older batch only), read whole but only two columns
    pick = sorted(rng.choice(len(files), size=min(n, len(files)), replace=False).tolist()) if files else []
    res['files'][kind] = len(pick)
    ids = pa.array(sorted(qmap.get(kind, {})), type=pa.string())
    for i in pick:
        t = pq.read_table(files[i], columns=['id', 'CreatedAt'])
        t = t.filter(pc.is_in(pc.cast(t['id'], pa.string()), value_set=ids))   # only rows that are also query texts
        if not t.num_rows: continue
        # no time shift here: the raw wall-clock value as epoch seconds
        ca = R.to_epoch_s(t['CreatedAt'])
        for aid, a in zip(t['id'].to_pylist(), ca.tolist()):
            u = qmap[kind][aid]
            # month key "YYYY-MM" of the query time; count this offset in that month
            by_month[str(np.datetime64(u, 's').astype('datetime64[M]'))][u - a] += 1
tot = collections.Counter()
for m in by_month.values(): tot.update(m)
n = sum(tot.values())
res['matched_rows'] = n
res['offsets_seconds_all'] = dict(tot.most_common(10))
res['offsets_seconds_by_month'] = {m: dict(by_month[m].most_common(4)) for m in sorted(by_month)}
res['share_8h'] = round(tot[28800] / n, 4) if n else None
res['share_7h'] = round(tot[25200] / n, 4) if n else None
# verdict: all offsets exactly 8 h -> fixed UTC-8; any 7 h offset -> daylight saving present; otherwise
# (neither) -> see table. Offsets other than 7 h / 8 h do not change the verdict once a 7 h offset exists.
res['verdict'] = ('no matches' if not n else 'fixed UTC-8 (the +8 h rule is right)' if tot[28800] == n else
                  'daylight saving present (7 h in summer months)' if tot[25200] > 0 else 'other offsets present; see table')
res['seconds'] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1))
