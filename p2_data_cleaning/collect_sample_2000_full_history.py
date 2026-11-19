"""Full, uncapped post + comment history for the 1,000 suspended / 1,000 active analysis sample.

Pulls every field Arctic Shift returns (no whitelist, no 10k cap) for each account in
p2_analysis/sample_2000_output/sample_accounts.parquet and writes raw JSON lines, one gzipped file per
account and stream: p2_data_cleaning/data/sample_2000_history/raw/<kind>/<username_key>.jsonl.gz

Large streams (> WINDOW_ITEMS expected items) are split into equal time windows that are pulled in
parallel and then concatenated (gzip members concatenate into one valid file). Window i covers
[start_i, end_i): `after` is exclusive, so a window starts at after = start_i - 1; `before` is exclusive.

Pagination inside a window (sort=asc, limit=auto):
- each next page starts at `after = last created_utc - 1` and rows are deduped on id, so the boundary
  second is re-read and nothing sharing a timestamp is skipped;
- a page with no new ids means either the end of the window or one second holding more items than a
  page. The second case (the whole page on one timestamp) is read from both ends; anything still
  unreachable is counted in `overflow_seconds`. Then `after = max ts` moves on and a following empty page
  ends the window. It never stops just because a page came back short.

In-pull checks, written to ledger.jsonl per window:
- order_violations: created_utc went backwards inside a page (the API's sort is tested, not assumed)
- author_mismatch: rows whose author is not this account
- outside_window: rows outside the window's [start, end)
- overflow_seconds: seconds with more items than could be paged
- a stalled cursor raises, and the window is recorded with its error

Never overwrites: finished windows and finished streams are skipped, so the run is resumable.
Usage: python collect_sample_2000_full_history.py [--limit N] [--workers N]
"""
import argparse
import gzip
import json
import math
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from requests_ratelimiter import LimiterSession

# Paths are relative to this script, so it runs from any folder.
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, '..', 'p2_analysis', 'sample_2000_output', 'sample_accounts.parquet')
OUT = os.path.join(HERE, 'data', 'sample_2000_history')
LEDGER = os.path.join(OUT, 'ledger.jsonl')
API = 'https://arctic-shift.photon-reddit.com/api'
KINDS = ('posts', 'comments')
START_UTC = 1104537600          # 2005-01-01, before Reddit existed; the API rejects after=0
WINDOW_ITEMS = 10_000           # expected items per time window for large streams
MAX_WINDOWS = 400

# Arctic Shift allows about 5 requests a second; all worker threads share this one session.
session = LimiterSession(per_second=5)
lock = threading.Lock()


class BadRequest(Exception):
    """A 400 from the API: the request itself is wrong, so retrying will not help."""


# One API call with up to 10 retries and exponential back-off; a 400 (bad request) fails at once.
def get(endpoint, **params):
    retry_count = 0
    while retry_count < 10:
        try:
            r = session.get(f'{API}/{endpoint}', params=params, timeout=120)
            body = r.json()
            if r.status_code == 200 and body.get('error') is None:
                return body['data'] or []
            if r.status_code == 400:
                raise BadRequest(f'400: {body.get("error")}')
            raise RuntimeError(f'{r.status_code}: {body.get("error")}')
        except BadRequest:
            raise
        except Exception as e:                     # includes non-JSON bodies (JSONDecodeError) from a struggling server
            retry_count += 1
            if retry_count >= 10:
                raise RuntimeError(f'{endpoint} {params} failed 10 times: {e}') from e
            time.sleep(min(2 ** retry_count, 120))


def pull_window(kind, author, path, start, end):
    """Every item by `author` with start <= created_utc < end, written to `path`."""
    seen = set()
    stats = dict(rows=0, pages=0, order_violations=0, author_mismatch=0, outside_window=0,
                 overflow_seconds=0, first_utc=None, last_utc=None)
    tmp = path + '.part'
    after = start - 1
    with gzip.open(tmp, 'wt', encoding='utf-8') as fh:
        # Write every not-yet-seen row to the window's file and update the in-pull checks; returns the new rows.
        def keep(rows):
            fresh = [r for r in rows if r['id'] not in seen]
            for r in fresh:
                seen.add(r['id'])
                t = int(r['created_utc'])
                stats['first_utc'] = t if stats['first_utc'] is None else min(stats['first_utc'], t)
                stats['last_utc'] = t if stats['last_utc'] is None else max(stats['last_utc'], t)
                if not start <= t < end:
                    stats['outside_window'] += 1
                if str(r.get('author', '')).lower() != author.lower():
                    stats['author_mismatch'] += 1
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            return fresh

        # Page oldest-first through the window until an empty page.
        while True:
            page = get(f'{kind}/search', author=author, sort='asc', limit='auto', after=after, before=end)
            stats['pages'] += 1
            if not page:
                break
            ts = [int(r['created_utc']) for r in page]
            if any(b < a for a, b in zip(ts, ts[1:])):
                stats['order_violations'] += 1
            fresh = keep(page)
            top = max(ts)
            if fresh:
                nxt = top - 1
            elif min(ts) == top:
                # one second fills the whole page: read it from both ends, then step past it
                keep(get(f'{kind}/search', author=author, sort='asc', limit=100, after=top - 1, before=top + 1))
                tail = get(f'{kind}/search', author=author, sort='desc', limit=100, after=top - 1, before=top + 1)
                if keep(tail) and len(tail) == 100:
                    stats['overflow_seconds'] += 1     # the end window was full and still new: the middle may be missing
                nxt = top
            else:
                nxt = top          # all seen: step past the last second; an empty page next ends the window
            if nxt < after or (nxt == after and not fresh):
                raise RuntimeError(f'cursor stalled at {after}')
            after = nxt

    stats['rows'] = len(seen)
    os.replace(tmp, path)
    return stats


# Append one window's result to ledger.jsonl (thread-safe).
def log(rec):
    with lock:
        with open(LEDGER, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')


# Pull one window and log the result; errors are recorded in the ledger, not raised.
def run_window(task):
    t0 = time.time()
    rec = {k: task[k] for k in ('username_key', 'author', 'kind', 'window', 'n_windows', 'start', 'end')}
    rec['error'] = None
    try:
        rec.update(pull_window(task['kind'], task['author'], task['path'], task['start'], task['end']))
    except Exception as e:
        rec['error'] = repr(e)[:500]
    rec['seconds'] = round(time.time() - t0, 1)
    rec['finished_utc'] = int(time.time())
    log(rec)
    return rec


# Final merged file of one stream.
def final_path(kind, key):
    return os.path.join(OUT, 'raw', kind, f'{key}.jsonl.gz')


# File of one time window of a stream, before merging.
def window_path(kind, key, i):
    return os.path.join(OUT, 'windows', kind, key, f'w{i:03d}.jsonl.gz')


# Concatenate a stream's window files into its final file once every window exists, then delete the windows.
def merge(kind, key, n):
    parts = [window_path(kind, key, i) for i in range(n)]
    if os.path.exists(final_path(kind, key)) or not all(os.path.exists(p) for p in parts):
        return False
    tmp = final_path(kind, key) + '.part'
    with open(tmp, 'wb') as out:
        for p in parts:
            with open(p, 'rb') as fh:
                shutil.copyfileobj(fh, out)
    os.replace(tmp, final_path(kind, key))
    shutil.rmtree(os.path.join(OUT, 'windows', kind, key))
    return True


def quick_get(endpoint, tries=4, **params):
    """Like get() but gives up fast: used only to size windows, never for data."""
    for i in range(tries):
        try:
            r = session.get(f'{API}/{endpoint}', params=params, timeout=120)
            body = r.json()
            if r.status_code == 200 and body.get('error') is None:
                return body['data'] or []
        except Exception:
            pass
        time.sleep(5 * (i + 1))
    return None


def window_edges(kind, author, expected, first, last, now):
    """Time edges that give windows of about WINDOW_ITEMS items each.

    Uses the aggregate endpoint's monthly counts when it answers; a month above WINDOW_ITEMS is cut into
    equal sub-periods. Otherwise falls back to equal cuts between the archive's first and last activity,
    plus one tail window to `now`. Edges only decide how work is split: windows always tile
    [START_UTC, now) exactly, so a poor split costs time, never data.
    """
    months = quick_get(f'{kind}/search/aggregate', author=author, aggregate='created_utc', frequency='month',
                       after=START_UTC, before=now)
    cuts = []
    if months:
        stamps = [int(pd.Timestamp(m['created_utc']).timestamp()) for m in months]
        counts = [int(m['count']) for m in months]
        acc = 0
        for i, (t0, c) in enumerate(zip(stamps, counts)):
            t1 = stamps[i + 1] if i + 1 < len(stamps) else now
            if c > WINDOW_ITEMS:
                k = math.ceil(c / WINDOW_ITEMS)
                cuts += [t0] + [t0 + (t1 - t0) * j // k for j in range(1, k)]
                acc = 0
            elif acc + c > WINDOW_ITEMS:
                cuts.append(t0)
                acc = c
            else:
                acc += c
    elif pd.notna(first) and pd.notna(last) and last > first:
        k = min(MAX_WINDOWS, math.ceil(expected / WINDOW_ITEMS))
        cuts = [int(first) + (int(last) - int(first)) * j // k for j in range(k)] + [int(last) + 1]
    cuts = sorted({c for c in cuts if START_UTC < c < now})
    if len(cuts) > MAX_WINDOWS - 1:                     # keep every k-th cut
        step = math.ceil(len(cuts) / (MAX_WINDOWS - 1))
        cuts = cuts[::step]
    return [START_UTC] + cuts + [now]


# One task per (account, stream, time window) that is not finished yet; small streams get one window.
def plan(accounts, now):
    tasks, n_windows = [], {}
    for r in accounts.itertuples():
        for kind in KINDS:
            if os.path.exists(final_path(kind, r.username_key)):
                continue
            expected = max(getattr(r, f'expected_{kind}'), getattr(r, f'n_{kind}'))
            edges = (window_edges(kind, r.arctic_arctic_author, expected, getattr(r, f'earliest_{kind[:-1]}_at'),
                                  getattr(r, f'arctic_last_{kind[:-1]}_at'), now)
                     if expected > WINDOW_ITEMS else [START_UTC, now])
            n = len(edges) - 1
            n_windows[(kind, r.username_key)] = n
            for i in range(n):
                path = window_path(kind, r.username_key, i)
                if os.path.exists(path):
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tasks.append(dict(username_key=r.username_key, author=r.arctic_arctic_author, kind=kind,
                                  window=i, n_windows=n, start=edges[i], end=edges[i + 1], path=path,
                                  expected=expected / n))
    # biggest windows first so the long tail does not run alone at the end
    tasks.sort(key=lambda t: -t['expected'])
    return tasks, n_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='only the first N accounts (smoke test)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    for kind in KINDS:
        os.makedirs(os.path.join(OUT, 'raw', kind), exist_ok=True)
# Columns of the 2,000-account sample: the archive's spelling of the username, expected counts and first/last activity.

    accounts = pd.read_parquet(SAMPLE, columns=['username_key', 'arctic_arctic_author', 'account_status',
                                                'expected_posts', 'expected_comments', 'n_posts', 'n_comments',
                                                'earliest_post_at', 'earliest_comment_at',
                                                'arctic_last_post_at', 'arctic_last_comment_at'])
    accounts = accounts.sort_values('username_key').reset_index(drop=True)
    if args.limit:
        accounts = accounts.head(args.limit)
    # the window end is fixed by the first run and reused on resume, so every stream shares one cut-off
    end_file = os.path.join(OUT, 'window_end_utc.txt')
    if not os.path.exists(end_file):
        with open(end_file, 'w') as fh:
            fh.write(str(int(time.time()) + 86_400))
    now = int(open(end_file).read())
    tasks, n_windows = plan(accounts, now)
    for (kind, key), n in n_windows.items():
        merge(kind, key, n)                       # streams whose windows all finished in an earlier run
    print(f'{len(accounts):,} accounts | {len(n_windows):,} streams to finish | {len(tasks):,} windows to pull '
          f'| window end {now}', flush=True)

    # Pull windows in parallel; merge a stream as soon as all its windows are done.
    t0, done, rows, errors = time.time(), 0, 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_window, t) for t in tasks]
        for f in as_completed(futures):
            rec = f.result()
            rows += rec.get('rows', 0)
            errors += rec['error'] is not None
            if rec['error'] is None:
                merge(rec['kind'], rec['username_key'], rec['n_windows'])
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f'  {done:,}/{len(tasks):,} windows | {rows:,} rows | {errors} errors | '
                      f'{(time.time() - t0) / 60:.1f} min', flush=True)
    left = sum(not os.path.exists(final_path(k, key)) for k, key in n_windows)
    print(f'done | streams still incomplete: {left}', flush=True)


if __name__ == '__main__':
    main()
