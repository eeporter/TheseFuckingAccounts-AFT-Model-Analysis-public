#!/usr/bin/env python3
"""Exact copy search (fastscan) with all current rules, 2026-11-17. One Slurm array task processes every NTASK-th file
of the selected archive files; row groups are spread over the task's cores (fork; index built once).

Query side = full copypasta candidate set (lsh/candidates_v1.parquet, 194,951 texts of the 987 model accounts),
  restricted by --query: 'pre2022' (created before 30 Nov 2022, default), 'preacc' (before the account's own accusation)
  or 'all'. The accusation time is carried into the output either way, so the stricter scope can be applied later.
Rules (both sides): game-post notice stripped first; markdown links reduced to their visible words and bare URLs removed;
  >= MINCHAR characters must remain and links may be at most half of the characters; then the Spark cleanup + word
  3-grams (fastscan.norm). Dupe = exact Jaccard >= 0.7.
Archive side: CreatedAt + 8 h = UTC; rows on/after 30 Nov 2022 dropped before any text work; newer batch never read.
A pair counts only if the archive item is strictly earlier than the query text (so self-matches drop out).
Writes ids, metadata and scores only; prints counts only.

Added notes (comment-only pass):
  Job: no recorded Slurm job in the README. It was tested by run_fast_test_v1.sh (also no recorded job). Its rule code
    (GAME/MDLINK/URL regexes, clean_and_keep, to_epoch_s) is the source that was copied into fastscan2/rules2.py.
  Inputs: $LSH_DIR/candidates_v1.parquet (query texts), $LSH_DIR/sample_accounts_dates_v1.parquet (accusation
    time per account key), and the archive parquet files under $ARCH_DIR (older batch only; --files can name files).
  Outputs (in --out): pairs_taskNNNN.parquet (one row per pair: query key, archive id/kind/author/subreddit,
    times, intersection, shingle counts, Jaccard) and summary_taskNNNN.json (counts and timings).
  Method: fastscan.Index (exact prefix filter + length filter + exact intersection, Jaccard >= 7/10); no Bloom
    filter in this version.
  Note: the markdown-link/URL reduction is used only for the length and link-share rule; the text that is
    normalised and shingled is the text with only the game notice removed (fastscan.norm then removes http(s) URLs
    and turns punctuation into spaces).
"""
import argparse, glob, json, os, re, sys, time
import multiprocessing as mp
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastscan as fs

ap = argparse.ArgumentParser()
ap.add_argument('--minchar', type=int, default=140)
ap.add_argument('--query', choices=['pre2022', 'preacc', 'all'], default='pre2022')
ap.add_argument('--files', default='older', help="'older' = all older-batch files, or a comma list of paths")
ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
ap.add_argument('--ntask', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
ap.add_argument('--out', required=True)
args = ap.parse_args()

L = os.environ.get('LSH_DIR', '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh')
ARCH = os.environ.get('ARCH_DIR', '/gscratch/comdata/output')
NEWER = ('9c8fe3bb', '0f597d44')                      # uuids of the 2025-26 batch (comments, submissions)
CUTOFF = int(np.datetime64('2022-11-30T00:00:00', 's').astype(np.int64))
NW = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
os.makedirs(args.out, exist_ok=True)
T = {'args': vars(args), 'workers': NW}; t0 = time.time()

# Rule regexes (RE2 via pyarrow):
#   GAME   - the "content not supported on old reddit" notice of game posts, with its optional link; removed first.
#   MDLINK - a markdown link [visible words](target); replaced by its visible words (group 1).
#   URL    - a bare URL starting http(s):// or www., up to the next ASCII whitespace.

GAME = r'(?i)this post contains content not supported on old reddit\.?\s*(\[click here to view the full post\]\([^)]*\))?'
MDLINK = r'\[([^\]]*)\]\([^)\s]*\)'
URL = r'(?i)(https?://|www\.)[^ \t\n\x0B\f\r]+'


def clean_and_keep(text):
    """text: string array. Returns (text with game notice removed, keep mask by length + link rule).

    keep = (visible characters after reducing markdown links to their words and removing bare URLs, trimmed,
    >= --minchar) AND (characters removed by that link reduction <= half of all characters), i.e.
    2 * (len(t) - len(nolink)) <= len(t). n_all is computed but not used.
    """
    t = pc.replace_substring_regex(pc.fill_null(text, ''), GAME, ' ')
    nolink = pc.replace_substring_regex(pc.replace_substring_regex(t, MDLINK, r'\1'), URL, ' ')
    n_all = pc.utf8_length(pc.utf8_trim_whitespace(t))
    n_vis = pc.utf8_length(pc.utf8_trim_whitespace(nolink))
    n_raw = pc.utf8_length(t)
    keep = pc.and_(pc.greater_equal(n_vis, args.minchar),
                   pc.less_equal(pc.multiply(pc.subtract(n_raw, pc.utf8_length(nolink)), 2), n_raw))
    return t, keep


def to_epoch_s(col):
    """Return a numpy int64 array of epoch seconds from a timestamp column (any unit) or a numeric column
    (cast to float, then truncated to int64)."""
    a = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if pa.types.is_timestamp(a.type):
        return pc.cast(pc.cast(a, pa.timestamp('s')), pa.int64()).to_numpy(zero_copy_only=False)
    return np.asarray(pc.cast(a, pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64).astype(np.int64)


# ---------- query side ----------
c = pq.read_table(f'{L}/candidates_v1.parquet', columns=['kind', 'id', 'outer_key', 'created_utc', 'text'])
# accusation time per account (first_mention_adj_utc, keyed by the same account key as outer_key in candidates);
# it is compared directly with q_created (epoch seconds) for --query preacc, so it must be epoch seconds too (the
# file's unit is not checked here); it is also written out as q_acc_utc
acc = pq.read_table(f'{L}/sample_accounts_dates_v1.parquet', columns=['username_key', 'first_mention_adj_utc']).to_pandas()
accmap = dict(zip(acc.username_key, acc.first_mention_adj_utc.astype(np.int64)))
T['candidates_total'] = c.num_rows
q_created = to_epoch_s(c['created_utc'])
q_acc = np.array([accmap.get(k, -1) for k in c['outer_key'].to_pylist()], dtype=np.int64)
assert (q_acc > 0).all(), 'missing accusation date'
# query scope: pre2022 = created before 30 Nov 2022 (CUTOFF); preacc = before the account's accusation; all = all
sel = q_created < CUTOFF if args.query == 'pre2022' else (q_created < q_acc if args.query == 'preacc' else np.ones(c.num_rows, bool))
qt, qkeep = clean_and_keep(c['text'])
sel &= qkeep.to_numpy(zero_copy_only=False)
c = c.filter(pa.array(sel)); qt = qt.filter(pa.array(sel)); q_created = q_created[sel]; q_acc = q_acc[sel]
T['queries'] = c.num_rows
T['query_accounts'] = len(set(c['outer_key'].to_pylist()))
# ckey = "kind:id", the query key written to the output
ckey = np.array([f'{k}:{i}' for k, i in zip(c['kind'].to_pylist(), c['id'].to_pylist())], dtype=object)
# build the prefix-filter inverted index once, in the parent; forked workers share it (threshold 7/10)
idx = fs.Index(fs.norm(qt), 7, 10)
del c, qt
T['index'] = {'query_shingles': idx.n_shingles, 'prefix_entries': idx.n_prefix}
MAXQ = int(q_created.max())                             # no archive row at/after this can be earlier than any query
LIMIT = min(CUTOFF, MAXQ)


def work(job):
    """Scan one archive row group. job = (parquet path, row-group index).

    Steps: read only the needed columns; shift CreatedAt by +8 h to UTC; drop rows at/after LIMIT; build the
    text (comment body, or post title + newline + selftext); apply clean_and_keep; normalise and scan against
    the index; keep only pairs where the archive item is strictly earlier than the query text.
    Returns (stats dict, pyarrow table of pairs). Runs in a forked worker, so idx.stats is that worker's copy.
    """
    path, rg = job
    t = time.time()
    kind = 'comment' if 'comments' in path else 'post'
    cols = ['id', 'author', 'subreddit', 'CreatedAt'] + (['body'] if kind == 'comment' else ['title', 'selftext'])
    tab = pq.ParquetFile(path).read_row_group(rg, columns=cols)
    n = tab.num_rows
    created = to_epoch_s(tab['CreatedAt']) + 8 * 3600     # archive CreatedAt is Pacific (8 h behind UTC)
    m_date = created < LIMIT
    tab = tab.filter(pa.array(m_date)); created = created[m_date]
    n_date = tab.num_rows
    text = tab['body'] if kind == 'comment' else pc.binary_join_element_wise(
        pc.fill_null(tab['title'], ''), pc.fill_null(tab['selftext'], ''), '\n')
    text, keep = clean_and_keep(text)
    keep = keep.to_numpy(zero_copy_only=False)
    tab = tab.filter(pa.array(keep)); text = text.filter(pa.array(keep)); created = created[keep]
    idx.stats[:] = 0
    r, q, inter, m, j = idx.scan(fs.norm(text))
    early = created[r] < q_created[q]                   # earlier-only rule (also removes self-matches)
    r, q, inter, m, j = r[early], q[early], inter[early], m[early], j[early]
    # r indexes rows of the filtered table (same order as text and created), so take() recovers their ids
    sub = tab.take(pa.array(r))
    hits = pa.table({'ckey': pa.array(ckey[q].astype(str)), 'aid': sub['id'], 'akind': pa.array([kind] * len(r)),
                     'a_author': sub['author'], 'a_subreddit': sub['subreddit'],
                     'a_created_utc': pa.array(created[r]), 'q_created_utc': pa.array(q_created[q]),
                     'q_acc_utc': pa.array(q_acc[q]), 'inter': pa.array(inter), 'a_nsh': pa.array(m),
                     'q_nsh': pa.array(idx.qlen[q]), 'jaccard': pa.array(j)})
    return dict(file=os.path.basename(path), rg=rg, kind=kind, rows=n, rows_date=n_date, rows_len=int(keep.sum()),
                verified=int(idx.stats[1]), pairs=len(r), cpu_s=time.time() - t), hits


# Main: list the older-batch files (both directories, minus the two newer-batch uuids), take every ntask-th file
# for this task, split them into (file, row group) jobs, run the jobs over NW forked workers, then write one pairs
# file and one summary per task. 'cpu_s' is wall time measured inside each worker, summed over workers.
if __name__ == '__main__':
    if args.files == 'older':
        files = sorted(f for d in ('reddit_comments_by_author.parquet', 'reddit_submissions_by_author.parquet')
                       for f in glob.glob(f'{ARCH}/{d}/*.parquet') if not any(u in f for u in NEWER))
    else:
        files = args.files.split(',')
    mine = files[args.task::args.ntask]
    T['files_total'] = len(files); T['files_this_task'] = len(mine)
    jobs = [(f, g) for f in mine for g in range(pq.ParquetFile(f).metadata.num_row_groups)]
    t1 = time.time(); stats = []
    parts = []
    with mp.get_context('fork').Pool(NW) as pool:
        for s, h in pool.imap_unordered(work, jobs, chunksize=1):
            stats.append(s)
            if h.num_rows: parts.append(h)
    if parts:
        pq.write_table(pa.concat_tables(parts), f'{args.out}/pairs_task{args.task:04d}.parquet')
    T['wall_s'] = round(time.time() - t1, 1)
    for k in ('rows', 'rows_date', 'rows_len', 'verified', 'pairs'):
        T[k] = int(sum(s[k] for s in stats))
    T['cpu_s'] = round(sum(s['cpu_s'] for s in stats), 1)
    T['rows_per_s_per_core'] = round(T['rows'] / max(T['cpu_s'], 1e-9))
    T['by_kind'] = {k: {x: int(sum(s[x] for s in stats if s['kind'] == k)) for x in ('rows', 'rows_date', 'rows_len', 'pairs')}
                    for k in ('comment', 'post')}
    T['queries_with_match'] = int(len(set().union(*[set(p['ckey'].to_pylist()) for p in parts]))) if parts else 0
    T['total_s'] = round(time.time() - t0, 1)
    json.dump(T, open(f'{args.out}/summary_task{args.task:04d}.json', 'w'), indent=1, default=str)
    print(json.dumps(T, indent=1, default=str))
