#!/usr/bin/env python3
"""Toy planted test for the LSH copy search (2026-11-12). Reads the first 5,000 rows of ONE archive comment file, samples 10 real comments
(20-300 words) and 3 short ones (3-7 words), and makes planted variants of them, plus 5 random-word negatives.
Writes toy_haystack_v1.parquet (those 5,000 rows), toy_cands_v1.parquet (cix, id, kind, variant, src_id, text) and toy_files_v1.json. Prints counts only."""
import glob, json, os, random, re
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
FILE = sorted(glob.glob('/gscratch/comdata/output/reddit_comments_by_author.parquet/*.parquet'))[0]
rng = random.Random(20261112)
pf = pq.ParquetFile(FILE)
first = pa.Table.from_batches([next(pf.iter_batches(batch_size=5_000))])                      # first 5,000 rows of the file: the toy haystack
pfull = first.to_pandas(timestamp_as_object=False)
d = pfull[['id', 'body']].copy()
d = d[~d.body.fillna('').str.strip().isin(['', '[deleted]', '[removed]'])]
tok = lambda s: re.sub(r"[^\w\s']", ' ', re.sub(r'https?://\S+', ' ', s.lower())).split()
d['ntok'] = d.body.map(lambda s: len(tok(s)))
src = d[(d.ntok >= 20) & (d.ntok <= 300)].sample(10, random_state=1)
short = d[(d.ntok >= 3) & (d.ntok <= 7)].sample(3, random_state=2)
vocab = sorted({w for s in src.body for w in tok(s)})
W = lambda s: s.split()
def swap(s, f):
    w = W(s)
    for i in rng.sample(range(len(w)), max(1, round(f * len(w)))): w[i] = rng.choice(vocab)
    return ' '.join(w)
def reformat(s): return ' '.join(x.upper() if rng.random() < .3 else x for x in W(s)).replace(' ', ' ,  ', 3) + ' !!!'
def trunc(s, f): w = W(s); return ' '.join(w[:max(1, round(f * len(w)))])
def append(s, f): w = W(s); o = W(src.body.iloc[rng.randrange(len(src))]); return ' '.join(w + o[:max(1, round(f * len(w)))])
V = {'source': lambda s: s, 'reformat': reformat, 'swap10': lambda s: swap(s, .10), 'swap50': lambda s: swap(s, .50),
     'trunc50': lambda s: trunc(s, .50), 'append25': lambda s: append(s, .25)}
rows = [(v, r.id, fn(r.body)) for r in src.itertuples() for v, fn in V.items()]
rows += [(v, r.id, t) for r in short.itertuples() for v, t in [('short_source', r.body), ('short_swap1', swap(r.body, .01))]]
rows += [('negative', None, ' '.join(rng.choice(vocab) for _ in range(rng.randint(20, 60)))) for _ in range(5)]
c = pd.DataFrame(rows, columns=['variant', 'src_id', 'text'])
c['id'] = [f'toy_{i:05d}' for i in range(len(c))]; c['kind'] = 'toy'; c['cix'] = np.arange(len(c), dtype='int64')
c.to_parquet(os.path.join(HERE, 'toy_cands_v1.parquet'), index=False)
# toy haystack = the same 5,000 rows (sources included); timestamps written as microseconds so Spark can read them
pq.write_table(first, os.path.join(HERE, 'toy_haystack_v1.parquet'), coerce_timestamps='us', allow_truncated_timestamps=True)
hay = first
json.dump({'archive_files': [os.path.join(HERE, 'toy_haystack_v1.parquet')], 'from_file': FILE},
          open(os.path.join(HERE, 'toy_files_v1.json'), 'w'))
print(len(hay), 'toy haystack rows')
print(len(c), 'toy candidates', c.variant.value_counts().to_dict())
