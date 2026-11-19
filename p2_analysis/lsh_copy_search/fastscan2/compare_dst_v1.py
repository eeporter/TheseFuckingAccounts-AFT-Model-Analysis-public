#!/usr/bin/env python3
"""compare_dst_v1.py OLD_OUT NEW_OUT - checks the daylight-saving rerun against the old run, streaming, counts only.

OLD_OUT was made with archive time = CreatedAt + 8 h (rules2/rules3); NEW_OUT with archive time = CreatedAt read as
America/Los_Angeles local time (rules4.archive_utc). Same queries, same archive, same matching. So:
  (a) every old pair must be in the new output, with a_created_utc equal to the old value minus the daylight-saving
      hour (or unchanged in standard time);
  (b) every new pair the old run did not have must be one the old time rule dropped: its old-rule archive time
      (local wall-clock + 8 h) is >= the query time, while its correct archive time is earlier. That can only happen
      for a daylight-saving archive time 0-1 h before the query.
How it is checked without holding billions of rows: for every NEW row the old-rule archive time is rebuilt from its
UTC time (UTC -> America/Los_Angeles wall clock, + 8 h). Rows where that is < q_created_utc are the ones the old run
must also have; a 64-bit hash of (ckey, aid, akind, inter, a_nsh, q_nsh, old-rule time) is summed over them (mod 2^64,
two different hash keys) and compared with the same sums over the OLD rows (with their stored a_created_utc).
Equal counts and equal sums = the same multiset of pairs (up to 64-bit hash collisions). The other new rows are
counted and must all have 0 < q_created_utc - a_created_utc <= 3600 and a daylight-saving archive time.
Exit 0 = all checks pass, 3 = a check fails.

Usage: compare_dst_v1.py OLD_OUT NEW_OUT [NPROC]   (NPROC = number of worker processes, default 4)
Inputs: every pairs_task*_p*.parquet in OLD_OUT and in NEW_OUT (the glob also matches worker-written
  pairs_task*_pw<pid>_*.parquet files); columns ckey, aid, akind, inter, a_nsh, q_nsh, a_created_utc, q_created_utc.
Outputs: one JSON object on stdout (file and pair counts, new_pairs_old_rule_also_keeps, new_only_pairs,
  new_only_not_dst_0_to_1h, old_pairs_all_in_new, PASS). Nothing is written to disk.
Jobs: checks_v4 43362085 step 2 (run_checks_v4.sh: OLD_OUT = full_out_v3, NEW_OUT = full_out_v4, NPROC = 8).

Caveat (not tested here): rebuilding the old-rule time assumes every archive CreatedAt is a real Pacific wall-clock
time. A CreatedAt inside the skipped March hour (02:00-02:59 on the change day), if any exists, is moved by
archive_utc, so its rebuilt old-rule time would differ from the stored one and the hash sums would not match."""
import sys, glob, json
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq

OLD, NEW = sys.argv[1], sys.argv[2]
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 4
# Hash keys for pandas' row hash. Two different keys give two independent 64-bit sums, so an accidental equality
# of both sums for different multisets is very unlikely.
KEYS = ('0123456789123456', '6543210987654321')          # two independent 16-byte hash keys
# pair identity columns hashed together with the (old-rule) archive time
COLS = ['ckey', 'aid', 'akind', 'inter', 'a_nsh', 'q_nsh']


def old_rule_time(a_utc):
    """UTC epoch seconds -> what the old rule computed: Pacific wall-clock time of that instant, + 8 h.

    This inverts rules4.archive_utc: the wall-clock time is recovered (UTC -> America/Los_Angeles, time zone then
    dropped) and the old flat + 8 h is applied. For a standard-time instant the result equals a_utc; for a
    daylight-saving instant it is a_utc + 1 h. Returns int64 epoch seconds."""
    local = pd.to_datetime(a_utc, unit='s', utc=True).tz_convert('America/Los_Angeles').tz_localize(None)
    return (local.to_numpy().astype('datetime64[s]').astype(np.int64) + 8 * 3600)


def hsum(df, tcol):
    """two order-free sums (mod 2^64) of a 64-bit row hash over COLS + the time column.

    Each row is hashed (pd.util.hash_pandas_object, one uint64 per row, one call per key in KEYS) and the hashes are
    summed with uint64 wrap-around. A sum does not depend on row order or on how rows are split across files, so the
    sums over all OLD rows and over the matching NEW rows are equal exactly when the two multisets of rows are equal
    (up to hash collisions). ckey, aid and akind are cast to str so both sides hash the same values."""
    d = df[COLS].astype({'ckey': str, 'aid': str, 'akind': str}).assign(t=tcol)
    return [int(pd.util.hash_pandas_object(d, index=False, hash_key=k).to_numpy(np.uint64).sum(dtype=np.uint64)) for k in KEYS]


def one(job):
    """Worker: one pair file. job = (side, path) with side 'old' or 'new'.

    Returns (side, dict): rows = all rows of the file; kept = rows included in the hash sums; h = the two hash sums
    of those rows; new_only = (new side) rows the old rule would have dropped; new_only_bad = new_only rows that are
    NOT a daylight-saving archive time 0-1 h before the query. Reads one row group at a time."""
    side, path = job
    pf = pq.ParquetFile(path)
    r = dict(rows=0, kept=0, h=[0, 0], new_only=0, new_only_bad=0)
    for g in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(g, columns=COLS + ['a_created_utc', 'q_created_utc']).to_pandas()
        r['rows'] += len(df)
        if side == 'old':
            # old run: its stored archive time is already the old-rule (+ 8 h) time; every row is hashed
            t = df['a_created_utc'].to_numpy(np.int64); keep = np.ones(len(df), bool)
        else:
            a = df['a_created_utc'].to_numpy(np.int64); q = df['q_created_utc'].to_numpy(np.int64)
            t = old_rule_time(a)                          # what the old run would have stored for this row
            keep = t < q                                  # the old run also has this pair
            off = t - 8 * 3600 - a                        # local - UTC: -7 h in daylight time, -8 h in standard
            gap = q - a
            no = ~keep
            r['new_only'] += int(no.sum())
            # a new-only pair is expected only if the archive item is 1 s to 1 h before the query AND its archive time is in
            # daylight time (local = UTC - 7 h); anything else is counted as bad
            r['new_only_bad'] += int((no & ~((gap > 0) & (gap <= 3600) & (off == -7 * 3600))).sum())
        if keep.any():
            hs = hsum(df[keep], t[keep])
            # add this row group's sums to the file's sums (mod 2^64, as Python ints)
            r['h'] = [(x + y) % 2 ** 64 for x, y in zip(r['h'], hs)]
        r['kept'] += int(keep.sum())
    return side, r


if __name__ == '__main__':
    # one job per pair file of either run; results are merged by side in whatever order the workers finish
    jobs = [('old', f) for f in sorted(glob.glob(f'{OLD}/pairs_task*_p*.parquet'))] + \
           [('new', f) for f in sorted(glob.glob(f'{NEW}/pairs_task*_p*.parquet'))]
    tot = {s: dict(files=0, rows=0, kept=0, h=[0, 0], new_only=0, new_only_bad=0) for s in ('old', 'new')}
    with Pool(NPROC) as p:
        for side, r in p.imap_unordered(one, jobs):
            T = tot[side]; T['files'] += 1
            for k in ('rows', 'kept', 'new_only', 'new_only_bad'): T[k] += r[k]
            T['h'] = [(x + y) % 2 ** 64 for x, y in zip(T['h'], r['h'])]
    o, n = tot['old'], tot['new']
    res = dict(old_files=o['files'], old_pairs=o['rows'], new_files=n['files'], new_pairs=n['rows'],
               new_pairs_old_rule_also_keeps=n['kept'], new_only_pairs=n['new_only'],
               new_only_not_dst_0_to_1h=n['new_only_bad'],
               # old_pairs_all_in_new: the new rows the old rule also keeps have the same count and the same two hash sums as all
               # old rows, i.e. the same multiset of (pair, old-rule archive time)
               old_pairs_all_in_new=(n['kept'] == o['rows'] and n['h'] == o['h']))
    # the last term is a bookkeeping check: every new row is either kept (old rule keeps it) or new-only
    res['PASS'] = res['old_pairs_all_in_new'] and res['new_only_not_dst_0_to_1h'] == 0 and n['rows'] == n['kept'] + n['new_only']
    print(json.dumps(res, indent=1))
    sys.exit(0 if res['PASS'] else 3)
