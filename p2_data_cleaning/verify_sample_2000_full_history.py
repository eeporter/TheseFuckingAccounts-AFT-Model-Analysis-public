"""Bug checks on the full sample history pulled by collect_sample_2000_full_history.py.

Per account and stream (posts, comments):
 1. pull_error      — a window ended with an error, so the stream has no merged file
 2. file integrity  — every line parses; id / created_utc / author present; no duplicate ids;
                      author matches the account; row count equals the ledger's window total
 3. vs aggregate    — pulled count equals Arctic Shift's independent aggregate count (exact)
 4. vs meta         — pulled count is at least the users/search _meta count (a stale lower bound),
                      and the first pulled item is the archive's stated earliest item
 5. vs old store    — every id in the earlier capped store (sample_2000_output) is in the new pull;
                      any missing id is fetched by id and written to recovered/<kind>.jsonl.gz
 6. reverse pull    — for a random subset of small streams, an independent newest-first pull must
                      return exactly the same ids
 7. empty windows   — for every time window that came back empty, the first item after its start
                      must fall at or after its end

Results go to checks/checks_<run utc>.parquet; nothing earlier is overwritten.
Usage: python verify_sample_2000_full_history.py [--reverse N]
"""
import argparse
import gzip
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# Reuses the paths, API helper and window logic of the pull script.
import collect_sample_2000_full_history as c

# Earlier capped store of the same sample, for check 5.
OLD = os.path.join(c.HERE, '..', 'p2_analysis', 'sample_2000_output')
CHECKS = os.path.join(c.OUT, 'checks')
RECOVERED = os.path.join(c.OUT, 'recovered')


# Check 2: read one merged stream file; count rows, duplicate ids, missing fields, bad lines and author mismatches.
def read_stream(kind, key, author):
    out = dict(rows=0, unique_ids=0, dup_ids=0, missing_fields=0, bad_lines=0, author_mismatch=0,
               first_utc=None, last_utc=None)
    ids = Counter()
    path = c.final_path(kind, key)
    if not os.path.exists(path):
        return out, set()
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                out['bad_lines'] += 1
                continue
            out['rows'] += 1
            if any(r.get(f) in (None, '') for f in ('id', 'created_utc', 'author')):
                out['missing_fields'] += 1
                continue
            ids[r['id']] += 1
            if str(r['author']).lower() != author.lower():
                out['author_mismatch'] += 1
            t = int(r['created_utc'])
            out['first_utc'] = t if out['first_utc'] is None else min(out['first_utc'], t)
            out['last_utc'] = t if out['last_utc'] is None else max(out['last_utc'], t)
    out['unique_ids'] = len(ids)
    out['dup_ids'] = sum(v - 1 for v in ids.values())
    return out, set(ids)


# Check 3: Arctic Shift's yearly aggregate count for the stream, independent of the paged pull.
def aggregate_count(kind, author, end):
    data = c.quick_get(f'{kind}/search/aggregate', tries=6, author=author, aggregate='created_utc',
                       frequency='year', after=c.START_UTC, before=end)
    return None if data is None else sum(int(m['count']) for m in data)


def reverse_ids(kind, author, end):
    """Independent newest-first pull: `before` is exclusive, so page with before = oldest ts + 1."""
    seen, before = set(), end
    while True:
        page = c.get(f'{kind}/search', author=author, sort='desc', limit='auto', after=c.START_UTC - 1, before=before)
        if not page:
            return seen
        ts = [int(r['created_utc']) for r in page]
        fresh = [r['id'] for r in page if r['id'] not in seen]
        seen.update(fresh)
        low = min(ts)
        nxt = low + 1 if fresh else low
        if nxt > before or (nxt == before and not fresh):
            raise RuntimeError(f'reverse cursor stalled at {before}')
        before = nxt


# Check 5: fetch items by id, 100 per request.
def fetch_by_id(kind, ids):
    got = []
    ids = sorted(ids)
    for i in range(0, len(ids), 100):
        got += c.get(f'{kind}/ids', ids=','.join(ids[i:i + 100]))
    return got


# Run checks 1-6 for one account; returns one result row per stream.
def check_account(args):
    r, end, old_ids, do_reverse = args
    rows = []
    meta = c.quick_get('users/search', tries=6, author=r.arctic_arctic_author)
    m = (meta[0].get('_meta') or {}) if meta else {}
    for kind in c.KINDS:
        rec = dict(username_key=r.username_key, account_status=r.account_status, kind=kind)
        rec['pull_error'] = not os.path.exists(c.final_path(kind, r.username_key))
        stats, ids = read_stream(kind, r.username_key, r.arctic_arctic_author)
        rec.update(stats)
        rec['aggregate_count'] = aggregate_count(kind, r.arctic_arctic_author, end)
        rec['meta_count'] = m.get(f'num_{kind}')
        rec['meta_earliest_utc'] = m.get(f'earliest_{kind[:-1]}_at')
        old = old_ids.get((kind, r.username_key), set())
        missing = old - ids
        rec['old_store_ids'] = len(old)
        rec['old_ids_missing'] = len(missing)
        rec['recovered_by_id'] = 0
        if missing:
            got = fetch_by_id(kind, missing)
            rec['recovered_by_id'] = len(got)
            path = os.path.join(RECOVERED, f'{kind}_{r.username_key}.jsonl.gz')
            if got and not os.path.exists(path):
                with gzip.open(path, 'wt', encoding='utf-8') as fh:
                    for x in got:
                        fh.write(json.dumps(x, ensure_ascii=False) + '\n')
        rec['reverse_checked'] = False
        rec['reverse_mismatch'] = None
        # Check 6 only for the randomly chosen accounts, and only for streams of at most 5,000 items.
        if do_reverse and 0 < len(ids) <= 5_000:
            rev = reverse_ids(kind, r.arctic_arctic_author, end)
            rec['reverse_checked'] = True
            rec['reverse_mismatch'] = len(rev ^ ids)
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reverse', type=int, default=100, help='accounts given the independent reverse pull')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(CHECKS, exist_ok=True)
    os.makedirs(RECOVERED, exist_ok=True)
    end = int(open(os.path.join(c.OUT, 'window_end_utc.txt')).read())

    accounts = pd.read_parquet(c.SAMPLE, columns=['username_key', 'arctic_arctic_author', 'account_status'])
    # Ids already held in the earlier capped store, per (stream, account).
    old_ids = {}
    for kind in c.KINDS:
        old = pd.read_parquet(os.path.join(OLD, f'sample_{kind}.parquet'), columns=['outer_key', 'id'])
        for key, g in old.groupby('outer_key'):
            old_ids[(kind, key)] = set(g['id'])
    # Fixed seed: the same accounts get the reverse pull on every run.
    random.seed(2026)
    reverse = set(random.sample(sorted(accounts['username_key']), args.reverse))

    jobs = [(r, end, old_ids, r.username_key in reverse) for r in accounts.itertuples()]
    t0, results = time.time(), []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, rows in enumerate(pool.map(check_account, jobs), 1):
            results += rows
            if i % 200 == 0:
                print(f'  {i:,}/{len(jobs):,} accounts checked | {(time.time() - t0) / 60:.1f} min', flush=True)

    checks = pd.DataFrame(results)
    # Add the ledger's row totals and in-pull check counts (last successful record per window).
    ledger = pd.read_json(os.path.join(c.OUT, 'ledger.jsonl'), lines=True)
    ok = ledger[ledger['error'].isna()].drop_duplicates(['username_key', 'kind', 'window'], keep='last')
    checks = checks.merge(ok.groupby(['username_key', 'kind'])['rows'].sum().rename('ledger_rows').reset_index(),
                          on=['username_key', 'kind'], how='left')
    for col in ('order_violations', 'overflow_seconds', 'outside_window'):
        checks = checks.merge(ok.groupby(['username_key', 'kind'])[col].sum().rename(f'pull_{col}').reset_index(),
                              on=['username_key', 'kind'], how='left')

    # 7. empty windows: the first item after an empty window's start must fall at or after its end
    empty = ok[(ok['rows'] == 0) & (ok['n_windows'] > 1)]
    def window_missed(w):
        page = c.get(f'{w.kind}/search', author=w.author, sort='asc', limit=1, after=int(w.start) - 1)
        return bool(page) and int(page[0]['created_utc']) < int(w.end)
    with ThreadPoolExecutor(args.workers) as pool:
        missed = list(pool.map(window_missed, empty.itertuples()))
    empty = empty.assign(missed=missed)
    checks = checks.merge(empty.groupby(['username_key', 'kind'])['missed'].sum().rename('empty_windows_missed')
                          .reset_index(), on=['username_key', 'kind'], how='left')
    print(f'empty windows re-probed: {len(empty):,}, with items inside: {int(empty["missed"].sum())}')

    # One yes/no flag per check; any_flag marks streams that need a look.
    checks['flag_integrity'] = ((checks[['dup_ids', 'missing_fields', 'bad_lines', 'author_mismatch']].sum(axis=1) > 0)
                                | (checks['rows'] != checks['ledger_rows'].fillna(0)))
    checks['flag_vs_aggregate'] = checks['aggregate_count'].notna() & (checks['unique_ids'] != checks['aggregate_count'])
    checks['flag_below_meta'] = checks['meta_count'].notna() & (checks['unique_ids'] < checks['meta_count'])
    checks['flag_first_item'] = (checks['meta_earliest_utc'].notna() & checks['first_utc'].notna()
                                 & (checks['first_utc'] != checks['meta_earliest_utc']))
    checks['flag_old_missing'] = checks['old_ids_missing'] > 0
    checks['flag_reverse'] = checks['reverse_mismatch'].fillna(0) > 0
    checks['flag_empty_window'] = checks['empty_windows_missed'].fillna(0) > 0
    checks['flag_pull'] = (checks['pull_error'] | (checks[['pull_order_violations', 'pull_overflow_seconds',
                                                            'pull_outside_window']].fillna(0).sum(axis=1) > 0))
    flags = [col for col in checks.columns if col.startswith('flag_')]
    checks['any_flag'] = checks[flags].any(axis=1)

    path = os.path.join(CHECKS, f'checks_{int(time.time())}.parquet')
    checks.to_parquet(path, engine='pyarrow')
    print(f'\nwrote {path}  {checks.shape}')
    print(f'streams: {len(checks):,}   rows pulled: {checks["unique_ids"].sum():,}')
    print(f'aggregate answered for {checks["aggregate_count"].notna().sum():,} streams; '
          f'reverse-checked {checks["reverse_checked"].sum():,} streams')
    print(checks[flags + ['any_flag']].sum().to_string())


if __name__ == '__main__':
    main()
