"""Current removal status of every pre-accusation pilot comment (AFT variable #19, extended model only).

Arctic Shift kept the original comment text, so removal is only visible through Reddit: PRAW shows a removed comment's
body as "[removed]" (and a self-deleted one as "[deleted]"). This is the status today, with no removal time, and it cannot
separate a moderator removal from a Reddit suspension sweep (user decision 2026-10-08: include in the extended model).

Writes numbered part files under comment_removal_output/parts/ (never overwrites); rerunning skips ids already done.
"""
import os, time, glob
import pandas as pd
import praw
from dotenv import load_dotenv

# Paths are relative to this script; the sample is the 998-account pilot. Reddit credentials come from a local .env file.
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, 'sample_998_v4_excl_output')
OUT = os.path.join(HERE, 'comment_removal_output', 'parts')
ENV_PATH = '/home/eeporter/Reddit_project1_notebooks/reddit_env_vars.env'
CHUNK = 5000  # comments per part file


# Write a part file, refusing to overwrite one that exists.
def save_new(df, path):
    if os.path.exists(path):
        raise FileExistsError(path)
    df.to_parquet(path, index=False)


# Removal status from the body Reddit shows today.
def body_status(body):
    if body == '[removed]':
        return 'removed'
    if body == '[deleted]':
        return 'deleted'
    return 'present'


# Look up up to 100 comments in one Reddit call; up to 10 tries with growing waits.
def lookup(reddit, ids):
    retry_count = 0
    while True:
        try:
            return {t.id: t for t in reddit.info(fullnames=['t1_' + i for i in ids])}
        except Exception:
            retry_count += 1
            if retry_count >= 10:
                raise
            time.sleep(min(60, 3 * retry_count))


def main():
    load_dotenv(ENV_PATH)
    reddit = praw.Reddit(client_id=os.getenv('praw_script'), client_secret=os.getenv('praw_key'),
                         username=os.getenv('login_name'), password=os.getenv('password'),
                         user_agent=os.getenv('user_agent'), ratelimit_seconds=300)
    os.makedirs(OUT, exist_ok=True)
    # Pre-accusation comments only: created before the account's edit-adjusted first accusation.
    a = pd.read_parquet(os.path.join(SAMPLE, 'sample_accounts.parquet'))[['username_key', 'first_mention_adj_utc']]
    c = pd.read_parquet(os.path.join(SAMPLE, 'sample_comments.parquet'), columns=['id', 'outer_key', 'created_utc'])
    c = c.drop_duplicates(['outer_key', 'id']).merge(a, left_on='outer_key', right_on='username_key')
    c = c[c.created_utc < c.first_mention_adj_utc][['id', 'outer_key']].reset_index(drop=True)
    # Resume: comment ids already in earlier part files are skipped.
    done_files = sorted(glob.glob(os.path.join(OUT, 'status_*.parquet')))
    done = set(pd.concat([pd.read_parquet(f) for f in done_files]).id) if done_files else set()
    c = c[~c.id.isin(done)]
    n_part = len(done_files)
    print(f'{len(done)} done, {len(c)} to go', flush=True)
    t0 = time.time()
    for start in range(0, len(c), CHUNK):
        chunk = c.iloc[start:start + CHUNK]
        rows = []
        # 100 comments per Reddit request; returned = False when Reddit no longer returns the comment.
        for b in range(0, len(chunk), 100):
            batch = chunk.iloc[b:b + 100]
            got = lookup(reddit, batch.id.tolist())
            now = int(time.time())
            for r in batch.itertuples(index=False):
                t = got.get(r.id)
                rows.append({'id': r.id, 'outer_key': r.outer_key, 'returned': t is not None,
                             'body_status': body_status(t.body) if t is not None else None,
                             'author_now': str(t.author) if t is not None and t.author is not None else None,
                             'checked_utc': now})
        n_part += 1
        save_new(pd.DataFrame(rows), os.path.join(OUT, f'status_{n_part:05d}.parquet'))
        k = start + len(chunk)
        print(f'{k}/{len(c)}  eta {(len(c) - k) * (time.time() - t0) / k / 60:.0f} min', flush=True)
    print('finished', flush=True)


if __name__ == '__main__':
    main()
