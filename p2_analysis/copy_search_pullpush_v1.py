"""Copied comments (#18, wider version, 2026-10-27): phrase-search each text in copy_search_output/search_list_v1.parquet
on PullPush (api.pullpush.io/reddit/search/comment, q = the comment text in quotes) and keep every match.

The query uses the first 200 characters of the comment's original text (URLs removed), in quotes. A comment counts as
copied if a match by a DIFFERENT author is at least as early as the comment. 3 queries at a time, with retries
(retry_count < 10, then re-raise). Writes numbered part files under copy_search_output/pullpush_parts/ (never
overwrites); rerunning skips ids already done.
"""
import os, re, glob, time, json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

# Paths are relative to this script. 3 parallel queries, 100 texts per part file, first 200 characters as the query.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'copy_search_output')
PARTS = os.path.join(OUT, 'pullpush_parts')
URL = 'https://api.pullpush.io/reddit/search/comment/'
WORKERS, CHUNK, QCHARS = 3, 100, 200
FIELDS = ['id', 'author', 'subreddit', 'created_utc', 'score', 'link_id', 'parent_id']


# Query text: URLs and punctuation removed, cut to 200 characters at a word boundary.
def query_text(body):
    t = re.sub(r'https?://\S+', ' ', body or '')
    t = re.sub(r"[^\w\s']", ' ', t)                            # punctuation breaks PullPush's query parser (400)
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:QCHARS].rsplit(' ', 1)[0] if len(t) > QCHARS else t


# Phrase-search one comment's text; up to 10 tries with growing waits on errors and rate limits.
def search(row):
    q = query_text(row.body)
    retry_count = 0
    while True:
        try:
            r = requests.get(URL, params={'q': f'"{q}"', 'size': 100}, timeout=90)
            if r.status_code == 429:
                raise RuntimeError('rate limited')
            if r.status_code == 400:                           # query the parser rejects: record, do not retry
                return {'id': row.id, 'outer_key': row.outer_key, 'created_utc': row.created_utc, 'query': q,
                        'n_hits': 0, 'hits': '[]', 'ok': False}
            r.raise_for_status()
            data = r.json().get('data', [])
            hits = [{k: d.get(k) for k in FIELDS} for d in data]
            return {'id': row.id, 'outer_key': row.outer_key, 'created_utc': row.created_utc, 'query': q,
                    'n_hits': len(hits), 'hits': json.dumps(hits), 'ok': True}
        except Exception:
            retry_count += 1
            if retry_count >= 10:
                raise
            time.sleep(min(60, 5 * 2 ** (retry_count - 1)))


def main():
    os.makedirs(PARTS, exist_ok=True)
    # The search list from copy_search_prep_v1.py, joined back to each comment's original text.
    todo = pd.read_parquet(os.path.join(OUT, 'search_list_v1.parquet'))
    comments = pd.read_parquet(os.path.join(HERE, 'sample_998_v4_excl_output', 'sample_comments.parquet'),
                               columns=['id', 'outer_key', 'body']).drop_duplicates(['outer_key', 'id'])
    todo = todo.drop(columns=[c for c in ['body'] if c in todo]).merge(comments, on=['outer_key', 'id'], how='left')   # original text
    # Resume: comment ids already in earlier part files are skipped.
    done_files = sorted(glob.glob(os.path.join(PARTS, 'part_*.parquet')))
    done = set(pd.concat([pd.read_parquet(f, columns=['id']) for f in done_files]).id) if done_files else set()
    todo = todo[~todo.id.isin(done)]
    print('to search', len(todo), flush=True)
    n = len(done_files)
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i in range(0, len(todo), CHUNK):
            chunk = todo.iloc[i:i + CHUNK]
            res = pd.DataFrame(list(ex.map(search, chunk.itertuples())))
            n += 1
            path = os.path.join(PARTS, f'part_{n:05d}.parquet')
            if os.path.exists(path):
                raise FileExistsError(path)
            res.to_parquet(path, index=False)
            print(f'{i + len(chunk)}/{len(todo)} searched, {round(time.time() - t0)} s', flush=True)


if __name__ == '__main__':
    main()
