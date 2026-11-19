#!/usr/bin/env python3
"""Copied comments (#18), Hyak version (2026-11-11): search the 2,162 texts in search_list_v1.parquet against every
comment in the CDSC Reddit archive (/gscratch/comdata/output/reddit_comments_by_author.parquet, through March 2026).

Replaces the PullPush phrase search (copy_search_pullpush_v1.py), which was rate limited. Same query: the comment's text
lower-cased, URLs removed, punctuation turned into spaces, whitespace collapsed, first 200 characters cut at a word
boundary. A hit is any archive comment whose text, normalised the same way, CONTAINS that query (Aho-Corasick over all
comments). The searched comment itself (same id) is kept, flagged is_self. No comment text is written out, only ids and metadata.

Usage: copy_search_hyak_v1.py TASK NTASKS   (the archive's files are split round-robin across tasks)
Writes hits_parts/file_<k>.parquet per archive file (one per file, even if empty); a rerun skips files already done,
so a preempted checkpoint job just resumes.

Method note: this is an older script that predates the word-3-gram / Jaccard / prefix-filter search
(Spark LSH, fastscan, fastscan2). It does NOT use shingles or Jaccard: a hit is an exact substring match of the
normalised, truncated query inside the normalised archive comment, found with one Aho-Corasick automaton over all
queries. Its normalisation (norm below) is its own: the plain RE2 word and whitespace classes, not the Spark cleanup.
Only comments are searched (no submissions). No date rule is applied here, and CreatedAt is copied as stored (no
+8 h shift); the earlier/later split is done afterwards by copy_search_combine_hyak_v1.py.

Inputs: search_list_v1.parquet next to this script (columns id, outer_key, created_utc, body); every *.parquet file in
  ARCHIVE (sorted; all files in that folder, no batch filter). Env BATCH = rows per Arrow batch (default 500,000).
Outputs: hits_parts/file_<k>.parquet, k = position of the archive file in the sorted list; columns id, outer_key,
  created_utc, query (the normalised query text) plus hit_* metadata of the archive comment, hit_len, is_self.
Job: Hyak Slurm array 43145502 (tasks 0-139, partition ckpt-g2) via copy_search_hyak_v1.sbatch, which runs
  `copy_search_hyak_v1.py $SLURM_ARRAY_TASK_ID 140`.
"""
import os, sys, glob, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import ahocorasick

# QCHARS: queries are cut to at most 200 characters; BATCH: archive rows read per batch (memory vs speed)
HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = '/gscratch/comdata/output/reddit_comments_by_author.parquet'
PARTS = os.path.join(HERE, 'hits_parts')
QCHARS, BATCH = 200, int(os.environ.get("BATCH", 500_000))
COLS = ['id', 'author', 'subreddit', 'CreatedAt', 'link_id', 'parent_id', 'score', 'body']


def norm(arr):
    """Normalise a string array: lower case, http(s) URLs -> space, every character that is not a word character,
    whitespace or apostrophe -> space, runs of whitespace -> one space, trim. Used on both queries and archive text.
    After this no text contains a newline, which the scan relies on."""
    a = pc.utf8_lower(arr)
    a = pc.replace_substring_regex(a, r'https?://\S+', ' ')
    a = pc.replace_substring_regex(a, r"[^\w\s']", ' ')
    a = pc.replace_substring_regex(a, r'\s+', ' ')
    return pc.utf8_trim_whitespace(a)


def queries():
    """Load the search list and build each query: norm(body), then, if longer than QCHARS characters, the first QCHARS
    characters cut back to the last space (so the query ends at a word boundary). Queries shorter than 20
    characters are dropped. Returns a DataFrame (id, outer_key, created_utc, query) with its original index."""
    q = pd.read_parquet(os.path.join(HERE, 'search_list_v1.parquet'), columns=['id', 'outer_key', 'created_utc', 'body'])
    q['query'] = norm(pa.array(q.body.fillna(''))).to_pylist()
    q['query'] = [t[:QCHARS].rsplit(' ', 1)[0] if len(t) > QCHARS else t for t in q['query']]
    q = q[q['query'].str.len() >= 20].drop(columns='body')
    return q


def build(q):
    """Aho-Corasick automaton over all distinct query strings. Each pattern's value is the list of DataFrame index
    labels of the queries with that exact text, so identical queries are matched once and reported for all."""
    A = ahocorasick.Automaton()
    for text, idx in q.groupby('query').groups.items():
        A.add_word(text, list(idx))
    A.make_automaton()
    return A


def scan(path, A, q):
    """Search one archive file. Returns a DataFrame with one row per (query, archive comment containing it).

    Per batch: normalise the body column, join all rows with '\n' into one string and run the automaton over it.
    Because neither queries nor normalised rows contain '\n', a match can never span two rows. The row of a match is
    found from the match's end position with a binary search over the positions of the row separators."""
    out = []
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=BATCH, columns=COLS):
        t = norm(b.column('body').fill_null(''))
        hay = '\n'.join(t.to_pylist())
        ends = np.cumsum(pc.utf8_length(t).to_numpy(zero_copy_only=False) + 1) - 1   # index of each row's '\n'
        rows, qidx = [], []
        # A.iter yields (index of the last character of the match, value = list of query index labels)
        for end, idxs in A.iter(hay):
            r = int(np.searchsorted(ends, end))
            rows.extend([r] * len(idxs)); qidx.extend(idxs)
        if rows:
            # archive metadata (all COLS except body) of the matching rows, prefixed hit_, beside the query columns
            meta = b.drop_columns(['body']).take(pa.array(rows)).to_pandas()
            meta.columns = ['hit_' + c for c in meta.columns]
            meta['hit_len'] = pc.utf8_length(t.take(pa.array(rows))).to_numpy(zero_copy_only=False)
            left = q.loc[qidx, ['id', 'outer_key', 'created_utc', 'query']].reset_index(drop=True)
            out.append(pd.concat([left, meta.reset_index(drop=True)], axis=1))
    if not out:
        return pd.DataFrame(columns=['id', 'outer_key', 'created_utc', 'query', 'hit_id', 'hit_author', 'hit_subreddit',
                                     'hit_CreatedAt', 'hit_link_id', 'hit_parent_id', 'hit_score', 'hit_len', 'is_self'])
    res = pd.concat(out, ignore_index=True)
    res['is_self'] = res.hit_id == res.id          # the searched comment itself: kept as a check that matching works
    # a query can occur more than once in one comment; keep one row per (query comment, archive comment)
    return res.drop_duplicates(['id', 'hit_id'])


def main():
    """Command line: TASK NTASKS. Task k handles archive files whose sorted position is k modulo NTASKS (round-robin).
    Each file's result is written to a .tmp file and renamed, so a file that exists is complete; existing files are
    skipped, which lets a requeued (preempted) task resume where it stopped."""
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(PARTS, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ARCHIVE, '*.parquet')))
    mine = [(k, f) for k, f in enumerate(files) if k % ntasks == task]
    q = queries()
    A = build(q)
    print(f'task {task}/{ntasks}: {len(mine)} of {len(files)} files, {len(q)} queries', flush=True)
    for k, f in mine:
        dest = os.path.join(PARTS, f'file_{k:05d}.parquet')
        if os.path.exists(dest):
            continue
        t0 = time.time()
        res = scan(f, A, q)
        res.to_parquet(dest + '.tmp', index=False)
        os.replace(dest + '.tmp', dest)
        print(f'file {k}: {len(res)} hits, {round(time.time() - t0)} s', flush=True)


if __name__ == '__main__':
    main()
