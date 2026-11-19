"""Full, uncapped post + comment history for the four ego-network case-study accounts.

Reuses the tested pagination and windowing from collect_sample_2000_full_history.py; writes raw JSON lines
(every field) to p2_data_cleaning/data/case_study_history/raw/<kind>/<account>.jsonl.gz and a run log
to ledger.jsonl. Finished streams are skipped, so it never overwrites and can resume.
Usage: python collect_case_study_full_history.py
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Reuse the tested pull functions of collect_sample_2000_full_history.py with a different output folder.
import collect_sample_2000_full_history as c

# The four ego-network case-study accounts.
ACCOUNTS = ['CaseA', 'CaseB', 'CaseC', 'CaseD']
c.OUT = os.path.join(c.HERE, 'data', 'case_study_history')
c.LEDGER = os.path.join(c.OUT, 'ledger.jsonl')


def main():
    for kind in c.KINDS:
        os.makedirs(os.path.join(c.OUT, 'raw', kind), exist_ok=True)
    # Fixed end time, written on the first run and reused on resume.
    end_file = os.path.join(c.OUT, 'window_end_utc.txt')
    if not os.path.exists(end_file):
        with open(end_file, 'w') as fh:
            fh.write(str(int(time.time())))
    end = int(open(end_file).read())

    tasks, n_windows = [], {}
    # Size the time windows from each account's Arctic Shift _meta counts, then queue unfinished windows.
    for author in ACCOUNTS:
        key = author.lower()
        meta = c.quick_get('users/search', tries=8, author=author)
        m = (meta[0].get('_meta') or {}) if meta else {}
        for kind in c.KINDS:
            if os.path.exists(c.final_path(kind, key)):
                continue
            expected = m.get(f'num_{kind}') or 0
            edges = (c.window_edges(kind, author, expected, m.get(f'earliest_{kind[:-1]}_at'),
                                    m.get(f'last_{kind[:-1]}_at'), end)
                     if expected > c.WINDOW_ITEMS else [c.START_UTC, end])
            n_windows[(kind, key)] = len(edges) - 1
            for i in range(len(edges) - 1):
                path = c.window_path(kind, key, i)
                if os.path.exists(path):
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tasks.append(dict(username_key=key, author=author, kind=kind, window=i, n_windows=len(edges) - 1,
                                  start=edges[i], end=edges[i + 1], path=path))
        print(f'{author}: meta posts {m.get("num_posts")}, comments {m.get("num_comments")}', flush=True)

    # Pull all windows, merge each stream, then print the row count per account and stream.
    with ThreadPoolExecutor(8) as pool:
        recs = list(pool.map(c.run_window, tasks))
    for (kind, key), n in n_windows.items():
        c.merge(kind, key, n)
    print(f'windows {len(recs)} | errors {sum(r["error"] is not None for r in recs)} | '
          f'rows {sum(r.get("rows", 0) or 0 for r in recs):,}', flush=True)
    for author in ACCOUNTS:
        for kind in c.KINDS:
            path = c.final_path(kind, author.lower())
            n = sum(1 for _ in __import__('gzip').open(path, 'rt')) if os.path.exists(path) else None
            print(f'  {author:20s} {kind:8s} {n}', flush=True)


if __name__ == '__main__':
    main()
