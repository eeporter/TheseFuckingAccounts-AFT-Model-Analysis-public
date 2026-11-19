"""Collect the replies each pilot account received before its accusation (AFT variable #8, and the inbound half of #17).

Two jobs, run in this order:
  posts    — every comment in the thread of each pre-accusation post that had comments
             (Arctic Shift comments/tree?link_id=, limit 10000 returns the whole thread in one request)
  comments — direct replies to each pre-accusation comment (comments/search?parent_id=t1_<id>, paged)

Each reply keeps its own created_utc, so the model can count replies only once they had arrived.
The 5 accounts with ~10,000 pre-accusation comments (archive cap) are skipped (user decision 2026-10-08).

Writes numbered part files under replies_output/<job>_parts/ (never overwrites); rerunning skips ids already done.
Usage: python collect_replies_v4.py [posts|comments|both]
"""
import os, sys, time, glob
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

# Paths are relative to this script; the sample is the 998-account pilot.
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, 'sample_998_v4_excl_output')
OUT = os.path.join(HERE, 'replies_output')
BASE = 'https://arctic-shift.photon-reddit.com/api'
WORKERS = 4
CHUNK = 500  # items per part file
# The 5 capped accounts skipped for this collection.
SKIP_ACCOUNTS = ['example-skip-a', 'exampleskipb', 'exampleskipc', 'example-skip-d', 'exampleskipe']
# Fields kept for every reply.
KEEP = ['id', 'name', 'link_id', 'parent_id', 'author', 'author_fullname', 'subreddit', 'created_utc', 'body',
        'edited', 'score', 'retrieved_on']


class BadRequest(Exception):
    pass


# One Arctic Shift request; a 400 fails at once, other errors are retried up to 10 times with waits up to 5 minutes.
def get(path, params):
    retry_count = 0
    while True:
        try:
            r = requests.get(BASE + path, params=params, timeout=180)
            if r.status_code == 400:
                raise BadRequest(f'400 {r.text[:200]}')
            j = r.json()
            if r.status_code == 200 and j.get('data') is not None:
                return j['data']
            raise RuntimeError(f'{r.status_code} {str(j)[:200]}')
        except BadRequest:
            raise
        except Exception:
            retry_count += 1
            if retry_count >= 10:
                raise
            time.sleep(min(300, 30 * retry_count))  # rides out Arctic Shift outages of ~20 min


# Flatten a comment tree into rows; 'more' stubs (replies the tree did not include) are counted, not followed.
def walk(nodes, out, more):
    for n in nodes:
        if n['kind'] == 'more':
            more.append(n['data'])
            continue
        d = n['data']
        out.append({k: d.get(k) for k in KEEP})
        r = d.get('replies')
        if isinstance(r, dict):
            walk(r['data']['children'], out, more)
        elif isinstance(r, list):
            walk(r, out, more)


# posts job: the whole comment thread under one pre-accusation post, with a count of replies hidden behind 'more' stubs.
def post_job(row):
    out, more = [], []
    walk(get('/comments/tree', {'link_id': row.id, 'limit': 10000}), out, more)
    for d in out:
        d['source_id'] = row.id
        d['outer_key'] = row.outer_key
    status = {'source_id': row.id, 'outer_key': row.outer_key, 'n_returned': len(out),
              'n_more_stubs': len(more), 'n_more_hidden': int(sum(m.get('count', 0) for m in more)),
              'collected_utc': int(time.time())}
    return out, status


# comments job: every direct reply to one pre-accusation comment, paged oldest first from 2005-01-01.
def comment_job(row):
    out, after = [], 1104537600
    while True:
        d = get('/comments/search', {'parent_id': 't1_' + row.id, 'limit': 100, 'after': after, 'sort': 'asc'})
        out.extend({k: x.get(k) for k in KEEP} for x in d)
        if not d:
            break
        # Next page starts one second before the last reply, so replies sharing that second are not skipped (deduped on id below).
        after = d[-1]['created_utc'] - 1
        if len(d) < 100:
            # confirm nothing lies past the last second before stopping
            more = get('/comments/search', {'parent_id': 't1_' + row.id, 'limit': 100, 'after': after + 1, 'sort': 'asc'})
            if not more:
                break
    out = list({d['id']: d for d in out}.values())
    for d in out:
        d['source_id'] = row.id
        d['outer_key'] = row.outer_key
    status = {'source_id': row.id, 'outer_key': row.outer_key, 'n_returned': len(out),
              'n_more_stubs': 0, 'n_more_hidden': 0, 'collected_utc': int(time.time())}
    return out, status


def save_new(df, path):
    if os.path.exists(path):
        raise FileExistsError(path)
    df.to_parquet(path, index=False)


# Items to collect: pre-accusation posts that had comments, or all pre-accusation comments, minus the skipped accounts.
def todo_items(job):
    a = pd.read_parquet(os.path.join(SAMPLE, 'sample_accounts.parquet'))[['username_key', 'first_mention_adj_utc']]
    a = a[~a.username_key.isin(SKIP_ACCOUNTS)]
    if job == 'posts':
        d = pd.read_parquet(os.path.join(SAMPLE, 'sample_posts.parquet'), columns=['id', 'outer_key', 'created_utc', 'num_comments'])
        d = d.drop_duplicates(['outer_key', 'id']).merge(a, left_on='outer_key', right_on='username_key')
        d = d[(d.created_utc < d.first_mention_adj_utc) & (d.num_comments.fillna(0) > 0)]
    else:
        d = pd.read_parquet(os.path.join(SAMPLE, 'sample_comments.parquet'), columns=['id', 'outer_key', 'created_utc'])
        d = d.drop_duplicates(['outer_key', 'id']).merge(a, left_on='outer_key', right_on='username_key')
        d = d[d.created_utc < d.first_mention_adj_utc]
    return d[['id', 'outer_key']].reset_index(drop=True)


# Run one job in chunks of CHUNK items, each written as a replies file plus a status file; skips items already done.
def run(job):
    parts = os.path.join(OUT, f'{job}_parts')
    os.makedirs(parts, exist_ok=True)
    done_files = sorted(glob.glob(os.path.join(parts, 'status_*.parquet')))
    done = set(pd.concat([pd.read_parquet(f) for f in done_files]).source_id) if done_files else set()
    items = todo_items(job)
    items = items[~items.id.isin(done)]
    n_part = len(done_files)
    fn = post_job if job == 'posts' else comment_job
    print(f'{job}: {len(done)} done, {len(items)} to go', flush=True)
    t0 = time.time()
    for start in range(0, len(items), CHUNK):
        chunk = list(items.iloc[start:start + CHUNK].itertuples(index=False))
        with ThreadPoolExecutor(WORKERS) as ex:
            res = list(ex.map(fn, chunk))
        n_part += 1
        replies = pd.DataFrame([r for out, _ in res for r in out], columns=KEEP + ['source_id', 'outer_key'])
        # edited is either False or a timestamp; stored as text so all part files share one type.
        replies['edited'] = replies['edited'].astype(str)
        status = pd.DataFrame([s for _, s in res])
        save_new(replies, os.path.join(parts, f'replies_{n_part:05d}.parquet'))
        save_new(status, os.path.join(parts, f'status_{n_part:05d}.parquet'))
        k = start + len(chunk)
        rate = (time.time() - t0) / k
        print(f'{job}: {k}/{len(items)}  {rate:.2f}s/item  eta {(len(items) - k) * rate / 3600:.1f} h', flush=True)
    print(f'{job}: finished', flush=True)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    for job in (['posts', 'comments'] if which == 'both' else [which]):
        run(job)
