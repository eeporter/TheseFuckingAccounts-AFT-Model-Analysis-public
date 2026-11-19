"""Fill the archive-retrieval times missing from the stored pilot history.

Arctic Shift stores the retrieval time under two names: older (Pushshift-era) items use `retrieved_on`, newer ones
`retrieved_utc`. The stored pull kept only `retrieved_utc` for posts and only `retrieved_on` for comments, so the other name
looked missing (50% of posts, 18% of comments). This re-reads those items by id (batches of 100) and keeps both names plus
the removal fields as Arctic Shift has them.

Writes numbered part files under retrieval_times_output/parts/ (never overwrites); rerunning skips ids already done.
"""
import os, time, glob
import pandas as pd
import requests

# Paths are relative to this script; the sample is the 998-account pilot.
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, 'sample_998_v4_excl_output')
OUT = os.path.join(HERE, 'retrieval_times_output', 'parts')
BASE = 'https://arctic-shift.photon-reddit.com/api'
CHUNK = 5000
# the API's `fields=` rejects whichever retrieval name a type lacks, so fetch whole records and keep these
FIELDS = {'posts': ['id', 'author', 'created_utc', 'retrieved_on', 'retrieved_utc', 'removed_by_category', 'score', 'edited'],
          'comments': ['id', 'author', 'created_utc', 'retrieved_on', 'retrieved_utc', 'score', 'body', 'edited']}


# Write a part file, refusing to overwrite one that exists.
def save_new(df, path):
    if os.path.exists(path):
        raise FileExistsError(path)
    df.to_parquet(path, index=False)


# Fetch up to 100 items by id and keep only the listed fields; up to 10 tries with growing waits.
def get(path, ids, fields):
    retry_count = 0
    while True:
        try:
            r = requests.get(BASE + path, params={'ids': ','.join(ids)}, timeout=120)
            j = r.json()
            if r.status_code == 200 and j.get('data') is not None:
                return [{k: d.get(k) for k in fields} for d in j['data']]
            raise RuntimeError(f'{r.status_code} {str(j)[:200]}')
        except Exception:
            retry_count += 1
            if retry_count >= 10:
                raise
            time.sleep(min(60, 3 * retry_count))


def main():
    os.makedirs(OUT, exist_ok=True)
    # Posts missing retrieved_utc and comments missing retrieved_on.
    todo = []
    p = pd.read_parquet(os.path.join(SAMPLE, 'sample_posts.parquet'), columns=['id', 'retrieved_utc'])
    todo.append(('posts', p[p.retrieved_utc.isna()].id.drop_duplicates()))
    c = pd.read_parquet(os.path.join(SAMPLE, 'sample_comments.parquet'), columns=['id', 'retrieved_on'])
    todo.append(('comments', c[c.retrieved_on.isna()].id.drop_duplicates()))
    for kind, ids in todo:
        # Resume: ids already in earlier part files are skipped.
        done_files = sorted(glob.glob(os.path.join(OUT, f'{kind}_*.parquet')))
        done = set(pd.concat([pd.read_parquet(f) for f in done_files]).id) if done_files else set()
        ids = ids[~ids.isin(done)].reset_index(drop=True)
        n_part = len(done_files)
        print(f'{kind}: {len(done)} done, {len(ids)} to go', flush=True)
        for start in range(0, len(ids), CHUNK):
            chunk = ids.iloc[start:start + CHUNK].tolist()
            rows = []
            for b in range(0, len(chunk), 100):
                rows.extend(get(f'/{kind}/ids', chunk[b:b + 100], FIELDS[kind]))
            got = pd.DataFrame(rows).drop_duplicates('id') if rows else pd.DataFrame(columns=['id'])
            # Keep every requested id; ids the archive no longer returns get empty fields.
            out = pd.DataFrame({'id': chunk}).merge(got, on='id', how='left')
            # Mixed-type columns stored as text so all part files share one schema.
            for col in ['edited', 'removed_by_category']:
                if col in out:
                    out[col] = out[col].astype(str)
            n_part += 1
            save_new(out, os.path.join(OUT, f'{kind}_{n_part:05d}.parquet'))
            print(f'{kind}: {start + len(chunk)}/{len(ids)}', flush=True)
    print('finished', flush=True)


if __name__ == '__main__':
    main()
