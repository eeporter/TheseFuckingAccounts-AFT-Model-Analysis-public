"""Data-integrity checks on the test-run daily covariates: timelines, ranges, and an independent recount for random account-days.

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../../aft_model/); kept for the record.
Checks: 1 timelines end at T and are contiguous from 0; 2 time origin is the earliest pre-accusation item; 3 accounts
near the 10,000-item cap; 4 flooding = 1 caused by a single thread; 5 recount of selected covariates for 300 random
account-days with activity, straight from the raw posts and comments.
Inputs: ../daily.parquet, pilot sample accounts/posts/comments. Output: integrity.csv next to this script.
Run: python integrity.py."""
import os, re, glob
import numpy as np, pandas as pd
P2 = '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_analysis'
S = os.path.join(P2, 'sample_998_v4_excl_output')
HERE = os.path.dirname(os.path.abspath(__file__))
DAY = 86400.0
daily = pd.read_parquet(os.path.join(HERE, '..', 'daily.parquet'))
acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
out = []
def rep(k, v):
    """Record check k with value v in `out` and print it."""
    out.append((k, v)); print(f'{k}: {v}', flush=True)

# 1. timelines
g = daily.groupby('id')
rep('accounts in daily', daily['id'].nunique())
T = acc['days_to_accused']
rep('max |last stop - days_to_accused|', float((g['stop'].max() - T.loc[g['stop'].max().index]).abs().max()))
rep('max |days_to_accused - (mention-first)/day|', float(((acc.first_mention_adj_utc - acc.first_action_utc) / DAY - T).abs().max()))
# each account starts at 0 and each row starts where the previous one stopped
rep('episodes start at 0 / contiguous', bool(g['start'].min().eq(0).all() and np.allclose(g['start'].shift(-1).dropna(), daily.loc[g['start'].shift(-1).notna(), 'stop'])))
rep('accounts accused within 1 day (no lagged behaviour possible)', int((T < 1).sum()))
rep('accounts accused within 2 days', int((T < 2).sum()))
rep('negative covariate values', int((daily.drop(columns=['id']).select_dtypes('number') < 0).sum().sum()))

# 2. time origin = earliest pre-accusation item
# (the `... if False else [...]` below always selects the second column list)
posts = pd.read_parquet(os.path.join(S, 'sample_posts.parquet'), columns=['id', 'outer_key', 'created_utc', 'subreddit', 'domain', 'is_self', 'url', 'selftext', 'score', 'over_18', 'link_flair_text'] if False else ['id', 'outer_key', 'created_utc', 'subreddit', 'domain', 'is_self', 'url', 'selftext', 'score', 'over_18']).drop_duplicates(['outer_key', 'id'])
coms = pd.read_parquet(os.path.join(S, 'sample_comments.parquet'), columns=['id', 'outer_key', 'created_utc', 'subreddit', 'link_id', 'body', 'score']).drop_duplicates(['outer_key', 'id'])
first = pd.concat([posts[['outer_key', 'created_utc']], coms[['outer_key', 'created_utc']]]).groupby('outer_key').created_utc.min()
d = (first.reindex(acc.index) - acc['first_action_utc'])
rep('accounts whose earliest item != first_action_utc (s)', int(d.ne(0).sum()))
rep('  differences (s) describe', d[d.ne(0)].describe().round(1).to_dict())
for x in (posts, coms):
    x['acc'] = x.outer_key.map(acc.first_mention_adj_utc); x['t0'] = x.outer_key.map(acc.first_action_utc)
rep('pre-accusation items before t0', int(((posts.created_utc < posts.t0) & (posts.created_utc < posts.acc)).sum() + ((coms.created_utc < coms.t0) & (coms.created_utc < coms.acc)).sum()))

# 3. history caps: accounts at the 10,000-item cap (history may be truncated at the start)
np_, nc_ = posts.groupby('outer_key').size(), coms.groupby('outer_key').size()
capped = sorted(set(np_[np_ >= 9900].index) | set(nc_[nc_ >= 9900].index))
rep('accounts with >= 9,900 posts or comments (possible cap)', capped)

# 4. flooding = 1 only because there was a single thread
tot_thr = daily.filter(like='thr_').sum(1)
rep('account-days with flooding == 1', int(daily.flooding.eq(1).sum()))
rep('  of which only one thread in the window', int((daily.flooding.eq(1) & tot_thr.eq(1)).sum()))

# 5. independent recount on 300 random account-days with activity
VIS = ('i.redd.it', 'v.redd.it', 'i.reddituploads.com', 'imgur.com', 'gfycat.com', 'redgifs.com', 'giphy.com', 'streamable.com', 'youtube.com', 'youtu.be')
posts = posts[posts.created_utc < posts.acc]; coms = coms[coms.created_utc < coms.acc]
for x in (posts, coms):
    x['day'] = np.floor((x.created_utc - x.t0) / DAY)
posts['score'] = pd.to_numeric(posts.score); coms['score'] = pd.to_numeric(coms.score)
rng = np.random.default_rng(7)
active = daily[(daily.filter(like='thr_').sum(1) > 0)]
samp = active.sample(300, random_state=3)
mism = {k: 0 for k in ['thr_total', 'down_posts', 'down_comments', 'popular_posts', 'has_posted', 'visual_posts_lower_bound']}
for r in samp.itertuples():
    # items in the 7 days before the account-day (the same 1-day-lagged window as build.py)
    p = posts[(posts.outer_key == r.id) & (posts.day >= r.start - 7) & (posts.day < r.start)]
    c = coms[(coms.outer_key == r.id) & (coms.day >= r.start - 7) & (coms.day < r.start)]
    thr = set(p.id) | set(c.link_id.str.removeprefix('t3_'))
    if len(thr) != sum(getattr(r, f'thr_{g}') for g in ['profile', 'adult', 'crypto', 'politics', 'karma_farming', 'general', 'news', 'sales', 'location', 'entertainment', 'other']):
        mism['thr_total'] += 1
    mism['down_posts'] += int(p.score.le(0).sum() != r.down_posts)
    mism['down_comments'] += int(c.score.lt(0).sum() != r.down_comments)
    mism['popular_posts'] += int(p.score.ge(100).sum() != r.popular_posts)
    allp = posts[(posts.outer_key == r.id) & (posts.day < r.start)]
    mism['has_posted'] += int(float(len(allp) > 0) != r.has_posted)
    # visual posts by link domain alone must not exceed visual_media (which also counts comments and text links)
    vis = p.domain.fillna('').str.lower().str.removeprefix('www.').apply(lambda h: any(h == v or h.endswith('.' + v) for v in VIS)).sum()
    mism['visual_posts_lower_bound'] += int(vis > r.visual_media)
rep('recount mismatches in 300 random active account-days', mism)
pd.DataFrame(out, columns=['check', 'value']).astype(str).to_csv(os.path.join(HERE, 'integrity.csv'), index=False)
