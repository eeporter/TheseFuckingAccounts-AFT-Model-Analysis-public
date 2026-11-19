#!/usr/bin/env python3
"""bruteforce_check_v2.py - independent check of a full-run output: EVERY query text against a random slice of archive
rows, counting all shingle overlaps exactly, with no prefix filter, no Bloom filter and no C code (counts only).

Changes from bruteforce_check_v1.py (18 Nov code review):
  * all queries are checked (v1: 1,000 random queries plus the queries with output pairs in the sampled row groups);
  * the archive side is a random slice of rows (--rows rows from --nrg-comment / --nrg-post random row groups, plus
    --big-rows rows from one row group the run itself flagged as having more than 2M pairs), so mass-copied rows are
    covered as well as ordinary ones;
  * the output pairs are collected for EVERY sampled archive row, including rows that fail the date or length / link
    rule, so a pair the run should not have written (archive row failing a rule) shows up as only_in_output. v1 only looked at
    rows that pass the rule, so it could not see such pairs.

Method: shingles are plain Python sets of "w1 w2 w3" strings from the reference cleanup (ref/fastscan.norm) applied to
the rule-checked text (rules2.clean_and_keep), as in v1. The archive slice gets an inverted index (shingle -> rows);
for each chunk of queries every (query, archive row) pair that shares at least one shingle gets its exact overlap count
c by expanding the postings and counting with numpy. A pair is kept when 10c >= 7(|Q|+|A|-c) (Jaccard >= 0.7) and the
archive row is strictly earlier (CreatedAt + 8 h < query created_utc). That set must equal the run's output pairs for
the sampled rows: same (query, archive row) pairs with the same c, |A|, |Q|.

Usage: bruteforce_check_v2.py --out FULL_OUT_DIR [--rows 50000 --nrg-comment 2 --nrg-post 1 --big-rows 2000 --seed 1]
Exit 0 = identical, 3 = differences found.

Inputs: LSH_DIR/candidates_v1.parquet + sample_accounts_dates_v1.parquet (query side, via rules2.Queries with --query
  and --minchar, default pre2022 / 50); ARCH_DIR archive files (older batch, or --files) from which the random row
  groups are drawn; --out = a full-run output folder: summary_task*.json (for big_row_groups) and every
  pairs_task*_p*.parquet (this glob also matches the worker-written pairs_task*_pw<pid>_*.parquet files).
Outputs: one JSON object on stdout (row groups used, sampled / rule-passing archive rows, queries checked, pair counts,
  only_in_bruteforce, only_in_output, IDENTICAL, seconds). Nothing is written to disk.
Jobs: checks_v3c 43361281 step 3 (run_checks_v3c.sh: --out full_out_v3, --rows 50000 --nrg-comment 2 --nrg-post 1
  --big-rows 2000 --seed 1, LSH_DIR = lsh_v2).

Independence from the search: shingles here are Python string sets (no xxhash64, no C code, no prefix filter, no
Bloom filter, no length filter), so a bug in any of those would show up as a difference. Shared with the search:
rules2 (Queries, clean_and_keep, date limit) and the reference cleanup ref/fastscan.norm."""
import argparse, glob, json, os, sys, collections, time
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'ref'))
import rules2 as R
import fastscan as ref                                  # reference cleanup (norm) only
# rules2 is used for the query selection (Queries), clean_and_keep, _text, _kind, to_epoch_s and NEWER. The run's
# rules3/rules4 copies of these functions are identical to rules2 apart from the archive time (and rules3's chunking).

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--rows', type=int, default=50000, help='random archive rows taken from the random row groups')
ap.add_argument('--nrg-comment', type=int, default=2)
ap.add_argument('--nrg-post', type=int, default=1)
ap.add_argument('--big-rows', type=int, default=2000, help='random rows from one row group listed in big_row_groups (0 = none)')
ap.add_argument('--qchunk', type=int, default=5000)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--minchar', type=int, default=50)
ap.add_argument('--query', default='pre2022')
ap.add_argument('--files', default='older', help="'older' or a comma list of files to sample row groups from")
args = ap.parse_args()
L = os.environ.get('LSH_DIR', '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh')
ARCH = os.environ.get('ARCH_DIR', '/gscratch/comdata/output')
t0 = time.time(); rng = np.random.default_rng(args.seed)
res = {'args': vars(args)}

files = sorted(f for d in ('reddit_comments_by_author.parquet', 'reddit_submissions_by_author.parquet')
               for f in glob.glob(f'{ARCH}/{d}/*.parquet') if not any(u in f for u in R.NEWER))
if args.files != 'older':
    files = args.files.split(',')
byname = {os.path.basename(f): f for f in files}

# ---- choose the archive slice: random row groups, then random rows inside them ----
# --nrg-comment random row groups from comment files and --nrg-post from submission files (files drawn with
# replacement, then one random row group per draw), plus one row group from the run's own big_row_groups list
units = []
for kind, n in (('comments', args.nrg_comment), ('submissions', args.nrg_post)):
    fk = [f for f in files if kind in f]
    for f in (rng.choice(fk, size=n, replace=True) if fk and n > 0 else []):
        units.append((str(f), int(rng.integers(pq.ParquetFile(f).metadata.num_row_groups)), 'random'))
big = []                                                # row groups the run itself flagged (> --chunk pairs)
for s in glob.glob(f'{args.out}/summary_task*.json'):
    big += [tuple(b[:2]) for b in json.load(open(s)).get('big_row_groups', []) if b[0] in byname]
if big and args.big_rows > 0:
    b = big[int(rng.integers(len(big)))]
    units.append((byname[b[0]], int(b[1]), 'big'))
# drop repeated (file, row group) picks, keeping the first
units = list(dict.fromkeys(units))
# recorded with a shortened file name (first 12 characters) only
res['row_groups'] = [(os.path.basename(f)[:12], g, why) for f, g, why in units]

Q = R.Queries(L, args.query, args.minchar)
res['queries_total'] = Q.n


def tri(s):
    """set of word 3-gram shingles of a normalised text (same definition as v1).

    s is already cleaned by ref.norm (lower case, URLs removed, non-word characters -> space, single spaces, trimmed),
    so splitting on ' ' gives the words. Each shingle is the string "w1 w2 w3" itself, not a hash. Texts with fewer
    than 3 words, or empty texts (split gives ['']), have no shingles."""
    w = s.split(' ')
    return {' '.join(w[i:i + 3]) for i in range(len(w) - 2)} if len(w) >= 3 and w[0] != '' else set()


# ---- archive rows: ids of ALL sampled rows (for collecting output pairs), shingle sets of rule-passing rows ----
a_kind, a_id, a_created, a_sets = [], [], [], []
all_ids = set()
n_rand_groups = sum(1 for u in units if u[2] == 'random')
for f, g, why in units:
    kind = R._kind(f)
    tab = pq.ParquetFile(f).read_row_group(g, columns=['id', 'CreatedAt'] + (['body'] if kind == 'comment' else ['title', 'selftext']))
    # archive time -> UTC as in the v3 run (rules2/rules3: flat + 8 h)
    created = R.to_epoch_s(tab['CreatedAt']) + 8 * 3600
    rows = np.arange(tab.num_rows)
    # rows to take: --big-rows from the big row group; --rows split evenly over the random row groups. Rows are drawn
    # without replacement and kept in file order.
    want = args.big_rows if why == 'big' else max(1, args.rows // max(n_rand_groups, 1))
    if len(rows) > want:
        rows = np.sort(rng.choice(rows, size=want, replace=False))
    tab = tab.take(pa.array(rows)); created = created[rows]
    ids = [str(x) for x in tab['id'].to_pylist()]
    all_ids.update((kind, i) for i in ids)              # every sampled row, even ones the date or length rule drops
    # t = notice-stripped text (links still in it, as in the search); keep = length + link rule
    t, keep, _ = R.clean_and_keep(R._text(tab, kind), Q.minchar)
    k = keep.to_numpy(zero_copy_only=False) & (created < Q.limit)   # same date mask and length / link rule as the search
    # only rows passing both rules get shingle sets; rule-failing rows stay in all_ids for the output side
    normed = ref.norm(t.filter(pa.array(k))).to_pylist()
    for i, c_, s in zip(np.array(ids, dtype=object)[k], created[k], normed):
        a_kind.append(kind); a_id.append(i); a_created.append(int(c_)); a_sets.append(tri(s))
res['archive_rows_sampled'] = len(all_ids); res['archive_rows_passing_rules'] = len(a_sets)
a_created = np.array(a_created, dtype=np.int64)
# a_n: |A|, number of distinct shingles of each rule-passing archive row
a_n = np.array([len(s) for s in a_sets], dtype=np.int64)

# inverted index of the archive slice: shingle id -> archive rows (CSR: post[ptr[v]:ptr[v+1]])
vocab = {}
# vocab: shingle string -> integer id, only for shingles seen in the archive slice. pv/pr: one (shingle id, row)
# entry per shingle of each row; sorting by shingle id gives the postings, cnt their lengths, ptr their start offsets.
pv, pr = [], []
for r, s in enumerate(a_sets):
    for x in s:
        pv.append(vocab.setdefault(x, len(vocab))); pr.append(r)
pv = np.array(pv, dtype=np.int64); pr = np.array(pr, dtype=np.int64)
o = np.argsort(pv, kind='stable'); post = pr[o]
cnt = np.bincount(pv, minlength=len(vocab)).astype(np.int64)
ptr = np.concatenate([[0], np.cumsum(cnt)])
NA = len(a_sets)

# ---- brute force: every query against every archive row of the slice ----
# For each chunk of --qchunk queries: look up every query shingle in vocab (shingles not in the slice cannot
# contribute), expand each hit into all archive rows of its postings, and count how often each (query, row) pair
# occurs. Because both sides are sets, that count is exactly c = |Q intersect A| for every pair with c >= 1.
# brute is a Counter (multiset) keyed by (ckey, aid, akind, c, |A|, |Q|) so it can be compared with the output.
brute = collections.Counter()
qn = 0
for q0 in range(0, Q.n, args.qchunk):
    q1 = min(Q.n, q0 + args.qchunk)
    sets = [tri(s) for s in ref.norm(Q.text.slice(q0, q1 - q0)).to_pylist()]
    qi, vi = [], []
    for j, s in enumerate(sets):
        for x in s:
            v = vocab.get(x)
            if v is not None: qi.append(j); vi.append(v)
    qn += len(sets)
    if not qi: continue
    qi = np.array(qi, dtype=np.int64); vi = np.array(vi, dtype=np.int64)
    ln = cnt[vi]                                        # postings length of each (query, shingle)
    tot = int(ln.sum())
    if tot == 0: continue
    rep_q = np.repeat(qi, ln)                           # expand: one entry per (query, shingle, archive row)
    # off: position 0..ln-1 inside each postings list, so post[ptr[v] + off] lists every row of shingle v
    off = np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln)
    rows = post[np.repeat(ptr[vi], ln) + off]
    # pair key = local query number * NA + archive row (unique per pair)
    key, c = np.unique(rep_q * NA + rows, return_counts=True)   # c = exact number of shared shingles
    jj = key // NA; rr = key % NA
    nq = np.array([len(sets[j]) for j in jj], dtype=np.int64)
    na = a_n[rr]
    qq = q0 + jj
    # exact Jaccard >= 0.7 in integers (10c >= 7(|Q|+|A|-c)) and the earlier-only rule (archive strictly earlier);
    # the date limit and length / link rule were already applied to the archive rows above, and to queries by Queries
    ok = (10 * c >= 7 * (nq + na - c)) & (a_created[rr] < Q.created[qq])
    for q, r, cc, a_, b_ in zip(qq[ok].tolist(), rr[ok].tolist(), c[ok].tolist(), na[ok].tolist(), nq[ok].tolist()):
        brute[(Q.ckey[q], a_id[r], a_kind[r], cc, a_, b_)] += 1
res['queries_checked'] = qn

# ---- the run's output pairs for EVERY sampled archive row (including rule-failing rows) ----
# The run's pairs whose archive row is any sampled row (passing the rules or not). Filtered first by aid alone (cheap,
# in Arrow), then by (akind, aid) so a comment and a post with the same id are not confused. Counted as a multiset,
# so a pair written twice by the run shows up as only_in_output.
outc = collections.Counter()
idset = pa.array(sorted({i for _, i in all_ids}), type=pa.string())
for fp in sorted(glob.glob(f'{args.out}/pairs_task*_p*.parquet')):
    pf = pq.ParquetFile(fp)
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=['ckey', 'aid', 'akind', 'inter', 'a_nsh', 'q_nsh'])
        t = t.filter(pc.is_in(pc.cast(t['aid'], pa.string()), value_set=idset))
        for r in zip(*[t[c].to_pylist() for c in t.column_names]):
            if (r[2], str(r[1])) in all_ids:
                outc[(r[0], str(r[1]), r[2], int(r[3]), int(r[4]), int(r[5]))] += 1
res['pairs_bruteforce'] = sum(brute.values()); res['pairs_output'] = sum(outc.values())
res['queries_with_pairs_here'] = len({k[0] for k in outc})
# Counter subtraction keeps positive counts only: only_in_bruteforce = pairs the run missed (or wrote with other
# counts); only_in_output = pairs the run wrote that brute force does not produce (including any pair of an archive
# row that fails a rule, or a duplicate row).
res['only_in_bruteforce'] = sum((brute - outc).values()); res['only_in_output'] = sum((outc - brute).values())
res['IDENTICAL'] = res['only_in_bruteforce'] == 0 and res['only_in_output'] == 0
res['seconds'] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1, default=str))
sys.exit(0 if res['IDENTICAL'] else 3)
