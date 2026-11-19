"""Text of the posts and comments each pilot account replied to before its accusation (AFT variable #18, option a:
a comment that copies the text it replied to). Same parents as collect_parent_authors_v4.py.

Every pre-accusation comment has a parent_id (t3_ = a post, t1_ = a comment). Arctic Shift's posts/ids and comments/ids
endpoints return the parent's title/selftext (posts) or body (comments) in batches of 100. All 998 accounts are included
(the 5 capped accounts were skipped only for the reply collection).

Writes numbered part files under parent_text_output/parts/ (never overwrites); rerunning skips ids already done.
"""
import os, time, glob
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
# Paths are relative to this script; the sample is the 998-account pilot.
SAMPLE = os.path.join(HERE, 'sample_998_v4_excl_output')
OUT = os.path.join(HERE, 'parent_text_output', 'parts')
BASE = 'https://arctic-shift.photon-reddit.com/api'
CHUNK = 5000  # parent ids per part file
# Fields requested: title and self-text for posts, body for comments.
FIELDS = {'/posts/ids': 'id,author,title,selftext', '/comments/ids': 'id,author,body'}


# Write a part file, refusing to overwrite one that exists.
def save_new(df, path):
    if os.path.exists(path):
        raise FileExistsError(path)
    df.to_parquet(path, index=False)


# Fetch up to 100 items by id from Arctic Shift; up to 10 tries with growing waits.
def get(path, ids):
    retry_count = 0
    while True:
        try:
            r = requests.get(BASE + path, params={'ids': ','.join(ids), 'fields': FIELDS[path]},
                             timeout=120)
            j = r.json()
            if r.status_code == 200 and j.get('data') is not None:
                return j['data']
            raise RuntimeError(f'{r.status_code} {str(j)[:200]}')
        except Exception:
            retry_count += 1
            if retry_count >= 10:
                raise
            time.sleep(min(60, 3 * retry_count))


def main():
    os.makedirs(OUT, exist_ok=True)
    # Pre-accusation comments of the pilot accounts (one row per account and comment id).
    a = pd.read_parquet(os.path.join(SAMPLE, 'sample_accounts.parquet'))[['username_key', 'first_mention_adj_utc']]
    c = pd.read_parquet(os.path.join(SAMPLE, 'sample_comments.parquet'), columns=['id', 'outer_key', 'created_utc', 'parent_id'])
    c = c.drop_duplicates(['outer_key', 'id']).merge(a, left_on='outer_key', right_on='username_key')
    c = c[c.created_utc < c.first_mention_adj_utc]
    # Each distinct parent once; skip parents already saved in earlier part files.
    parents = pd.Series(c.parent_id.dropna().unique())
    done_files = sorted(glob.glob(os.path.join(OUT, 'parents_*.parquet')))
    done = set(pd.concat([pd.read_parquet(f) for f in done_files]).parent_id) if done_files else set()
    parents = parents[~parents.isin(done)].sort_values().reset_index(drop=True)
    n_part = len(done_files)
    print(f'{len(done)} done, {len(parents)} to go', flush=True)
    t0 = time.time()
    for start in range(0, len(parents), CHUNK):
        chunk = parents.iloc[start:start + CHUNK]
        rows = []
        # t3_ parents are posts, t1_ parents are comments; ids are sent without the prefix.
        for prefix, path in [('t3_', '/posts/ids'), ('t1_', '/comments/ids')]:
            ids = [p[3:] for p in chunk if p.startswith(prefix)]
            for b in range(0, len(ids), 100):
                for d in get(path, ids[b:b + 100]):
                    rows.append({'parent_id': prefix + d['id'], 'parent_author': d.get('author'),
                                 'parent_title': d.get('title'), 'parent_text': d.get('selftext', d.get('body'))})
        got = pd.DataFrame(rows, columns=['parent_id', 'parent_author', 'parent_title', 'parent_text'])
        got = got.drop_duplicates('parent_id')
        # Keep every parent id; returned = False when the archive no longer has it.
        out = pd.DataFrame({'parent_id': chunk.values}).merge(got, on='parent_id', how='left')
        out['returned'] = out.parent_author.notna()
        n_part += 1
        save_new(out, os.path.join(OUT, f'parents_{n_part:05d}.parquet'))
        k = start + len(chunk)
        print(f'{k}/{len(parents)}  eta {(len(parents) - k) * (time.time() - t0) / k / 60:.0f} min', flush=True)
    print('finished', flush=True)


if __name__ == '__main__':
    main()
