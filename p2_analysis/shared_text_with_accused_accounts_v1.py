"""Shares text with another accused account (user, 2026-10-27): for every post and comment of the pilot accounts, does
an EARLIER (or same-second) post or comment by a DIFFERENT accused account have the same text?

Comparison set: the full stored Arctic Shift histories of all accused, not-deleted accounts (p2_data_cleaning/data/
arctic_posts, arctic_comments; read file by file, de-duplicated on outer_key + id). Text is normalised as in
copy_search_pullpush_prep_v1.py (URLs and markdown quote lines removed, lower case, whitespace collapsed); posts use title +
self-text. Only texts of 60+ characters count, so short generic lines ("thank you!") do not. Exact match on the
normalised text. Writes shared_text_output/shared_text_flags_v1.parquet (one row per pilot item; never overwrites).
"""
import os, re, glob, time
import numpy as np
import pandas as pd

# Paths are relative to this script. DATA = the stored Arctic Shift histories of all accused accounts; S = the 998-account pilot.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'p2_data_cleaning', 'data')
S = os.path.join(HERE, 'sample_998_v4_excl_output')
OUT = os.path.join(HERE, 'shared_text_output')
MIN_CHARS = 60
GONE = {'[removed]', '[deleted]'}


# Normalise text: URLs and markdown quote lines removed, lower case, whitespace collapsed.
def norm(s):
    s = s.fillna('').astype(str).str.replace(r'https?://\S+', ' ', regex=True)
    s = s.str.replace(r'^\s*&gt;.*$', ' ', regex=True, flags=re.M)
    return s.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()


# Post text = title + self-text (removed/deleted bodies count as empty).
def post_text(d):
    body = d.selftext.where(~d.selftext.isin(GONE), '')
    return norm(d.title.fillna('') + ' ' + body.fillna(''))


def comment_text(d):
    return norm(d.body.where(~d.body.isin(GONE), ''))


# 64-bit hash of each normalised text; equal texts get equal hashes, so matching is a join on h.
def hashed(t):
    return pd.util.hash_pandas_object(t, index=False).to_numpy()


# --- 1. Hash every 60+ character text of every accused account, file by file ---
t0 = time.time()
parts = []
for kind, cols, fn in [('posts', ['id', 'outer_key', 'created_utc', 'title', 'selftext'], post_text),
                       ('comments', ['id', 'outer_key', 'created_utc', 'body'], comment_text)]:
    for f in sorted(glob.glob(os.path.join(DATA, f'arctic_{kind}', '*.parquet'))):
        d = pd.read_parquet(f, columns=cols)
        if not len(d):
            continue
        txt = fn(d)
        keep = (txt.str.len() >= MIN_CHARS).to_numpy()
        parts.append(pd.DataFrame({'outer_key': d.outer_key.to_numpy()[keep], 'id': d.id.to_numpy()[keep],
                                   'utc': pd.to_numeric(d.created_utc).to_numpy()[keep], 'h': hashed(txt[keep])}))
    print(kind, 'read', round(time.time() - t0), 's', flush=True)
allx = pd.concat(parts, ignore_index=True).drop_duplicates(['outer_key', 'id'])
print('comparison items (60+ chars):', len(allx), 'accounts', allx.outer_key.nunique(), flush=True)

# first use of each text by each account; number of accounts that ever used it
first = allx.sort_values('utc').drop_duplicates(['h', 'outer_key'])
n_acc = first.groupby('h').outer_key.size()

# --- 2. Hash the pilot accounts' own posts and comments ---
posts = pd.read_parquet(os.path.join(S, 'sample_posts.parquet'), columns=['id', 'outer_key', 'created_utc', 'title', 'selftext']).drop_duplicates(['outer_key', 'id'])
coms = pd.read_parquet(os.path.join(S, 'sample_comments.parquet'), columns=['id', 'outer_key', 'created_utc', 'body']).drop_duplicates(['outer_key', 'id'])
items = pd.concat([pd.DataFrame({'kind': 'post', 'outer_key': posts.outer_key, 'id': posts.id, 'utc': pd.to_numeric(posts.created_utc), 'text': post_text(posts)}),
                   pd.DataFrame({'kind': 'comment', 'outer_key': coms.outer_key, 'id': coms.id, 'utc': pd.to_numeric(coms.created_utc), 'text': comment_text(coms)})],
                  ignore_index=True)
items = items[items.text.str.len() >= MIN_CHARS].copy()
items['h'] = hashed(items.text)
# Comparison texts that also occur in the pilot, grouped by hash: (accounts, times).
cand = first[first.h.isin(set(items.h))].sort_values('utc')
g = {h: (grp.outer_key.to_numpy(), grp.utc.to_numpy()) for h, grp in cand.groupby('h')}
none = (np.array([]), np.array([]))
earliest_other, n_other_before = [], []
# For each pilot item: the earliest same-text use by a DIFFERENT account at or before it, and how many such accounts.
for k, u, h in zip(items.outer_key.to_numpy(), items.utc.to_numpy(), items.h.to_numpy()):
    who, when = g.get(h, none)
    m = (who != k) & (when <= u)
    earliest_other.append(when[m].min() if m.any() else np.nan)
    n_other_before.append(len(set(who[m])))
items['earliest_other_utc'] = earliest_other
items['n_other_accounts_before'] = n_other_before
items['shared'] = items.n_other_accounts_before > 0
items['n_accounts_any_time'] = items.h.map(n_acc).fillna(1).astype(int)
out = items.drop(columns=['text', 'h'])
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'shared_text_flags_v1.parquet')
if os.path.exists(path):
    raise FileExistsError(path)
out.to_parquet(path, index=False)
print('pilot items 60+ chars', len(out), 'shared', int(out.shared.sum()), 'accounts with any', out[out.shared].outer_key.nunique(),
      '| by kind', out.groupby('kind').shared.sum().to_dict(), round(time.time() - t0), 's')
