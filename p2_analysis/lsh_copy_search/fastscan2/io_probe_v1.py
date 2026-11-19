#!/usr/bin/env python3
"""io_probe_v1.py - READ-ONLY disk test for the copy search, 2026-11-18. Prints byte counts and times only.

Question: does keeping more reads in flight make reading the archive faster? Each "arm" uses N reader processes to
read the exact bytes the copy search reads per row group (the CreatedAt and body column chunks of a comment file,
one whole column chunk per read, as pyarrow does). Nothing is decoded, nothing is written except a counts-only JSON.

Arms run one after another (default 4, 12, 24, 48, then 4 again to see if the disk changed during the job). Every arm
reads its OWN row groups: max(--rgs-per-arm, 2 x readers) of them, taken from the next unused files in the sorted list of
older-batch comment files starting at position --offset, so every reader does at least 2 reads. No row group is read
twice in this job, so no arm is helped by an earlier arm. The JSON is rewritten after every arm.
Usage: io_probe_v1.py --offset 600 --rgs-per-arm 48 --arms 4,12,24,48,4 --out io_probe_v1.json

Input: older-batch comment files (*074ac9d9*) under ARCH_DIR (env, default /gscratch/comdata/output); only parquet
metadata is parsed, then raw bytes are read with os.pread. Output: --out JSON (per arm: readers, GB, wall time,
MB/s, median and p90 read time) plus a speedup of every arm relative to the first.
Job: run_io_probe_v1.sh, job 43332089 (4/12/24/48 readers, then 4 again).
Files from sorted position --stop-before (620) on are never read: they were kept unread (uncached) for the
4-vs-12 worker A/B test (run_ab_workers_v1.sh uses positions 620-623). The assert fires once the NEXT position would
reach --stop-before, so in practice the last file this can read is position stop_before - 2."""
import argparse, glob, json, os, socket, time
import multiprocessing as mp
import numpy as np, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument('--offset', type=int, default=600)
ap.add_argument('--rgs-per-arm', type=int, default=48)
ap.add_argument('--stop-before', type=int, default=620, help='never read sorted file positions at or after this (kept for the A/B test)')
ap.add_argument('--arms', default='4,12,24,48,4')
ap.add_argument('--out', required=True)
args = ap.parse_args()
ARCH = os.environ.get('ARCH_DIR', '/gscratch/comdata/output')
COLS = ('CreatedAt', 'body')

files = sorted(glob.glob(f'{ARCH}/reddit_comments_by_author.parquet/*074ac9d9*.parquet'))   # older-batch comments only
arms = [int(x) for x in args.arms.split(',')]


def ranges(path, n_rg):
    """(offset, length) of the CreatedAt and body column chunks of the first n_rg row groups.
    Returns a list of (path, row group, [(offset, length), ...]) units. A chunk starts at its dictionary page if it has
    one, else at its first data page; total_compressed_size is the whole chunk on disk."""
    md = pq.ParquetFile(path).metadata
    names = [md.schema.column(i).path for i in range(md.num_columns)]
    ci = [names.index(c) for c in COLS]
    out = []
    for g in range(min(n_rg, md.num_row_groups)):
        rg = md.row_group(g); u = []
        for i in ci:
            c = rg.column(i)
            start = c.dictionary_page_offset if (c.has_dictionary_page and c.dictionary_page_offset) else c.data_page_offset
            u.append((start, c.total_compressed_size))
        out.append((path, g, u))
    return out


def read_unit(unit):
    """Read the column chunks of one row group with unbuffered os.pread (no decoding). Returns (bytes read, seconds)."""
    path, g, u = unit
    t0 = time.time(); n = 0
    with open(path, 'rb', buffering=0) as f:
        for off, ln in u:
            got = 0
            while got < ln:                                   # one read per column chunk; loop only if the OS returns less
                b = os.pread(f.fileno(), ln - got, off + got)
                assert b, 'unexpected end of file'
                got += len(b)
            n += got
    return n, time.time() - t0


if __name__ == '__main__':
    R = {'host': socket.gethostname(), 'cpus': os.environ.get('SLURM_CPUS_PER_TASK'), 'files_total': len(files),
         'offset': args.offset, 'columns': COLS, 'arms': []}
    T0 = time.time(); pos = args.offset
    for k, nr in enumerate(arms):
        need = max(args.rgs_per_arm, 2 * nr); units = []; fl = []
        while len(units) < need:                              # whole files only, each file used by one arm only
            fl.append(files[pos]); units += ranges(files[pos], 10 ** 6); pos += 1
            assert pos < args.stop_before, 'would reach files kept for the A/B test'
        units = units[:need]
        with mp.get_context('fork').Pool(nr) as pool:
            # pool.map(abs, ...) is a no-op task that makes every reader process exist before t0
            pool.map(abs, range(nr))                          # start all readers before the clock starts
            t0 = time.time()
            res = list(pool.imap_unordered(read_unit, units, chunksize=1))
            wall = time.time() - t0
        nb = sum(r[0] for r in res); ts = np.array([r[1] for r in res]); per = np.array([r[0] / r[1] / 1e6 for r in res])
        a = {'arm': k, 'readers': nr, 't_start_s': round(t0 - T0, 1), 'file_positions': [pos - len(fl), pos - 1], 'row_groups': len(units),
             'GB': round(nb / 1e9, 3), 'wall_s': round(wall, 2), 'MB_per_s': round(nb / wall / 1e6, 1),
             'read_s_median': round(float(np.median(ts)), 2), 'read_s_p90': round(float(np.percentile(ts, 90)), 2),
             'MB_per_s_per_read_median': round(float(np.median(per)), 1)}
        R['arms'].append(a); print(json.dumps(a), flush=True)
        json.dump(R, open(args.out, 'w'), indent=1)
    base = R['arms'][0]['MB_per_s']
    R['speedup_vs_first_arm'] = {f"{a['arm']}:{a['readers']}": round(a['MB_per_s'] / base, 2) for a in R['arms']}
    R['DONE'] = True
    json.dump(R, open(args.out, 'w'), indent=1)
    print(json.dumps(R['speedup_vs_first_arm']))
