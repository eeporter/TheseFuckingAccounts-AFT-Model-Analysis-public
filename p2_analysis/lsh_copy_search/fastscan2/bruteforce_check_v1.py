#!/usr/bin/env python3
"""bruteforce_check_v1.py - independent check of a full-run output on a sample (counts only).

Takes a few archive row groups (chosen at random with a fixed seed) and a set of query texts, and finds every pair
with Jaccard >= 0.7 WITHOUT the prefix filter, the Bloom filter or any C code: shingles are plain Python sets of
"w1 w2 w3" strings (no hashing), and every query's whole shingle set is counted against every archive row
(an exact all-overlaps count, so every pair with any shared shingle is scored). It uses only the rule code
(rules2.clean_and_keep, the date and earlier-only rules) and the reference cleanup (ref/fastscan.norm).
The pairs it finds must equal the run's output for those queries and row groups: same pairs, same counts.

Queries checked: --nq random queries plus every query that has an output pair in the chosen row groups.
Usage: bruteforce_check_v1.py --out FULL_OUT_DIR [--nrg-comment 2 --nrg-post 1 --nq 1000 --seed 1 --minchar 50]
Exit 0 = identical, 3 = differences found.

Inputs: --out = full-run output folder (pairs_task*_p*.parquet); LSH_DIR (env) = query folder for rules2.Queries;
ARCH_DIR (env) = archive root; --files = 'older' (all older-batch files, R.NEWER excluded) or an explicit file list.
Output: one counts-only JSON on stdout.
Jobs: check 5 of run_checks_v3.sh (job 43340007) and run_checks_v3b.sh (job 43360112), both with LSH_DIR=lsh_v2,
--nrg-comment 2 --nrg-post 1 --nq 1000 --seed 1; also staged in run_scale_test_v1.sh (never submitted).

Why this is independent of the scanner: no hashing (shingles are the "w1 w2 w3" strings themselves, so no xxhash64
collision can hide or create a match), no prefix filter, no Bloom filter, no length filter, no fastscan2 C code.
The archive rows go through the reference pipeline: date mask, +8 h, clean_and_keep, ref.norm. Every pair whose
Jaccard >= 0.7 (integer test 10c >= 7(|A|+|B|-c)) and whose archive item is strictly earlier is kept.
Comparison: both sides as multisets (collections.Counter) of (ckey, aid, akind, inter, a_nsh, q_nsh).
Note: with n >= 10**6 for a kind (test mode) every row group of that kind in the file list is used."""
import argparse, glob, json, os, sys, collections, time
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'ref'))
import rules2 as R
import fastscan as ref                                  # reference cleanup (norm) only

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--nrg-comment', type=int, default=2)
ap.add_argument('--nrg-post', type=int, default=1)
ap.add_argument('--nq', type=int, default=1000)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--minchar', type=int, default=50)
ap.add_argument('--query', default='pre2022')
ap.add_argument('--files', default='older', help="'older' or a comma list: sample row groups only from these files (use the files the run scanned)")
args = ap.parse_args()
L = os.environ.get('LSH_DIR', '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh')
ARCH = os.environ.get('ARCH_DIR', '/gscratch/comdata/output')
t0 = time.time(); rng = np.random.default_rng(args.seed)
res = {'args': vars(args)}

# candidate archive files: older batch only (files of the 2025-26 batch, R.NEWER, are skipped), or the --files list
files = sorted(f for d in ('reddit_comments_by_author.parquet', 'reddit_submissions_by_author.parquet')
               for f in glob.glob(f'{ARCH}/{d}/*.parquet') if not any(u in f for u in R.NEWER))
if args.files != 'older':
    files = args.files.split(',')
units = []
for kind, n in (('comments', args.nrg_comment), ('submissions', args.nrg_post)):
    fk = [f for f in files if kind in f]
    if not fk or n <= 0: continue
    if n >= 10 ** 6:                                    # test mode: every row group of this kind
        units += [(f, g) for f in fk for g in range(pq.ParquetFile(f).metadata.num_row_groups)]; continue
    for f in rng.choice(fk, size=n, replace=True):      # a random file, then a random row group in it
        units.append((str(f), int(rng.integers(pq.ParquetFile(f).metadata.num_row_groups))))
    units = list(dict.fromkeys(units))                  # drop row groups drawn twice (keeps first-drawn order)
res['row_groups'] = [(os.path.basename(f)[:12], g) for f, g in units]

Q = R.Queries(L, args.query, args.minchar)
res['queries_total'] = Q.n


def tri(s):
    """word 3-gram shingles of one cleaned text, as a set of plain strings "w1 w2 w3" (no hashing).
    Fewer than 3 words, or an empty text, gives the empty set (such a text has no shingles and can match nothing)."""
    w = s.split(' ')
    return {' '.join(w[i:i + 3]) for i in range(len(w) - 2)} if len(w) >= 3 and w[0] != '' else set()


# archive rows of the chosen row groups: ids, times, shingle sets (reference pipeline, no index)
arch = []
for f, g in units:
    kind = R._kind(f)
    tab = pq.ParquetFile(f).read_row_group(g, columns=['id', 'CreatedAt'] + (['body'] if kind == 'comment' else ['title', 'selftext']))
    # archive CreatedAt is stored 8 h behind UTC (Pacific): +8 h before any date comparison
    created = R.to_epoch_s(tab['CreatedAt']) + 8 * 3600
    # Q.limit = min(30 Nov 2022 cutoff, latest query time): rows at/after it can never be earlier than any query
    m = created < Q.limit
    tab = tab.filter(pa.array(m)); created = created[m]
    # game notice stripped; length + link rule (>= minchar visible characters, links at most half)
    t, keep, vis = R.clean_and_keep(R._text(tab, kind), Q.minchar)
    k = keep.to_numpy(zero_copy_only=False)
    ids = tab['id'].filter(pa.array(k)).to_pylist(); created = created[k]
    normed = ref.norm(t.filter(pa.array(k))).to_pylist()
    arch.append((kind, ids, created, [tri(s) for s in normed]))
aid_sets = {(kind, i) for kind, ids, _, _ in arch for i in ids}
res['archive_rows_checked'] = sum(len(a[1]) for a in arch)

# output pairs that fall in these row groups
out_rows = []
for fp in sorted(glob.glob(f'{args.out}/pairs_task*_p*.parquet')):
    t = pq.read_table(fp, columns=['ckey', 'aid', 'akind', 'inter', 'a_nsh', 'q_nsh'])
    # cheap prefilter on aid alone, then exact (akind, aid) membership below
    t = t.filter(pc.is_in(t['aid'], value_set=pa.array(sorted({i for _, i in aid_sets}))))
    out_rows += [r for r in zip(*[t[c].to_pylist() for c in t.column_names]) if (r[2], r[1]) in aid_sets]
ck = {c: i for i, c in enumerate(Q.ckey.tolist())}
# queries to check: --nq random ones plus every query that has an output pair in these row groups (so every output
# pair here is re-derived, and random queries test for pairs the run may have missed)
qsel = set(int(x) for x in rng.choice(Q.n, size=min(args.nq, Q.n), replace=False)) | {ck[r[0]] for r in out_rows if r[0] in ck}
res['queries_checked'] = len(qsel); res['queries_with_output_pairs_here'] = len({r[0] for r in out_rows})
qsel = sorted(qsel)

# brute force: every selected query against every archive row (all overlaps counted exactly)
qnorm = ref.norm(Q.text.take(pa.array(qsel))).to_pylist()
qsets = [tri(s) for s in qnorm]
# plain inverted index over each selected query's WHOLE shingle set (not a prefix): string shingle -> query positions
inv = collections.defaultdict(list)
for j, s in enumerate(qsets):
    for x in s: inv[x].append(j)
brute = []
for kind, ids, created, sets in arch:
    for r, s in enumerate(sets):
        if not s: continue
        # cnt[j] = exact intersection size c with query j (each shingle of a set is counted once)
        cnt = collections.Counter(j for x in s for j in inv.get(x, ()))
        na = len(s)
        for j, c in cnt.items():
            nq = len(qsets[j]); q = qsel[j]
            # Jaccard >= 0.7 in integers, and the earlier-only rule (archive strictly before the query text)
            if 10 * c >= 7 * (nq + na - c) and created[r] < Q.created[q]:
                brute.append((Q.ckey[q], ids[r], kind, c, na, nq))
# output pairs are compared only for the checked queries
qkeys = {Q.ckey[q] for q in qsel}
outc = collections.Counter(r for r in out_rows if r[0] in qkeys)
brc = collections.Counter(brute)
res['pairs_bruteforce'] = sum(brc.values()); res['pairs_output'] = sum(outc.values())
res['only_in_bruteforce'] = sum((brc - outc).values()); res['only_in_output'] = sum((outc - brc).values())
res['IDENTICAL'] = res['only_in_bruteforce'] == 0 and res['only_in_output'] == 0
res['seconds'] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1, default=str))
sys.exit(0 if res['IDENTICAL'] else 3)
