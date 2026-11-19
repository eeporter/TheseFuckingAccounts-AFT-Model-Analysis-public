"""Exact-time build: covariates change at the moment of each action (and when it leaves the 7-day window), not on a
daily grid with a 1-day lag. Flooding needs 2+ threads. Accounts whose capped archive ends before the accusation are
censored at the archive end. Data loading is shared with build.py."""
import os, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'build.py')).read().split("GROUPS = ['profile'")[0]
exec(src)                                   # acc, groups, posts, coms (pre-accusation, flagged), rep, tfa, DAY, WIN
GROUPS = ['profile', 'adult', 'crypto', 'politics', 'karma_farming', 'general', 'news', 'sales', 'location', 'entertainment', 'other']
a_full = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')

# censoring: capped stream (>= 10,000 items, history_truncated) whose last archived item is before the accusation
cens = {}
allp = pd.read_parquet(os.path.join(S, 'sample_posts.parquet'), columns=['outer_key', 'id', 'created_utc']).drop_duplicates(['outer_key', 'id'])
allc = pd.read_parquet(os.path.join(S, 'sample_comments.parquet'), columns=['outer_key', 'id', 'created_utc']).drop_duplicates(['outer_key', 'id'])
trunc = a_full.index[a_full.history_truncated.fillna(False).astype(bool)]
for d in (allp, allc):
    g = d[d.outer_key.isin(trunc)].groupby('outer_key').created_utc.agg(['size', 'max'])
    for k, r in g[g['size'] >= 10000].iterrows():
        if r['max'] < acc.loc[k, 'first_mention_adj_utc']:
            c = (r['max'] - acc.loc[k, 'first_action_utc']) / DAY
            cens[k] = min(c, cens.get(k, np.inf))
print('censored at archive end:', {k: round(v, 1) for k, v in cens.items()}, flush=True)

for d in (posts, coms):
    d['t'] = (d.created_utc - d.t0) / DAY
    e = pd.to_numeric(d.edited.replace('False', np.nan), errors='coerce')
    d['edit_t'] = np.where(e.notna() & (e < d['acc_utc']) & (e >= d.t0), (e - d.t0) / DAY, np.nan)
rep['t'] = (pd.to_numeric(rep.created_utc) - rep.t0) / DAY


def window(times, grid):
    """items with time in (a - WIN, a] at each grid point a"""
    s = np.sort(np.asarray(times, float))
    return np.searchsorted(s, grid, 'right') - np.searchsorted(s, grid - WIN, 'right')


def thread_active(td, grid):
    """td: DataFrame(thread, t) -> number of threads with an item in (a - WIN, a] at each grid point a"""
    if td.empty:
        return np.zeros(len(grid), int)
    td = td.sort_values(['thread', 't'])
    th, t = td.thread.to_numpy(), td.t.to_numpy()
    new_block = np.r_[True, (th[1:] != th[:-1]) | (np.diff(t) > WIN)]
    bid = np.cumsum(new_block)
    s = pd.Series(t).groupby(bid).agg(['min', 'max'])
    starts, ends = np.sort(s['min'].to_numpy()), np.sort(s['max'].to_numpy() + WIN)
    return np.searchsorted(starts, grid, 'right') - np.searchsorted(ends, grid, 'right')


t0c = time.time()
rows = []
pg, cg, rg = posts.groupby('outer_key'), coms.groupby('outer_key'), rep.groupby('outer_key')
for k, a in acc.iterrows():
    T = a.days_to_accused
    end = min(T, cens.get(k, np.inf))
    p = pg.get_group(k) if k in pg.groups else posts.iloc[:0]
    c = cg.get_group(k) if k in cg.groups else coms.iloc[:0]
    r = rg.get_group(k) if k in rg.groups else rep.iloc[:0]
    p, c, r = p[p.t < end], c[c.t < end], r[r.t < end]
    times = np.r_[p.t, c.t, p.edit_t.dropna(), c.edit_t.dropna(), r.t]
    pts = np.r_[0.0, times, times + WIN, np.arange(1, np.ceil(end))]
    act = pd.concat([p[['thread', 't', 'group', 'subreddit']], c[['thread', 't', 'group', 'subreddit']]])
    if len(act):                                  # thread-block ends also change the thread counts
        pts = np.r_[pts, act.groupby('thread').t.max().to_numpy() + WIN]
    grid = np.unique(pts[(pts >= 0) & (pts < end)])
    f = {'id': np.repeat(k, len(grid)), 'start': grid, 'stop': np.r_[grid[1:], end], 'event': np.r_[np.zeros(len(grid) - 1), float(k not in cens)]}
    for gname in GROUPS:
        f[f'thr_{gname}'] = thread_active(act[act.group == gname], grid)
    tot = np.zeros(len(grid)); mx = np.zeros(len(grid))
    for _, s in act.groupby('subreddit'):
        w = thread_active(s, grid); tot += w; mx = np.maximum(mx, w)
    f['flooding'] = np.where(tot >= 2, mx / np.maximum(tot, 1), 0.0)
    f['links_external'] = window(np.r_[p.t[p.external], c.t[c.external]], grid)
    f['visual_media'] = window(np.r_[p.t[p.visual], c.t[c.visual]], grid)
    f['crossposts'] = window(p.t[p.crosspost], grid)
    f['nsfw_outside'] = window(p.t[p.nsfw_outside], grid)
    f['edits'] = window(np.r_[p.edit_t.dropna(), c.edit_t.dropna()], grid)
    f['down_posts'] = window(p.t[p.down], grid)
    f['down_comments'] = window(c.t[c.down], grid)
    f['popular_posts'] = window(p.t[p.popular], grid)
    f['popular_comments'] = window(c.t[c.popular], grid)
    f['replies_on_posts'] = window(r.t, grid)
    f['has_posted'] = (grid >= (p.t.min() if len(p) else np.inf)).astype(float)
    f['has_commented'] = (grid >= (c.t.min() if len(c) else np.inf)).astype(float)
    cal = a.first_action_utc + np.floor(grid) * DAY             # TFA control stays a daily calendar measure
    f['tfa_posts_30d'] = np.searchsorted(tfa, cal) - np.searchsorted(tfa, cal - 30 * DAY)
    rows.append(pd.DataFrame(f))
ex = pd.concat(rows, ignore_index=True)
print(f'episodes {len(ex)}  accounts {ex.id.nunique()}  censored {int((ex.groupby("id").event.last() == 0).sum())}  {time.time() - t0c:.0f}s', flush=True)
ex.to_parquet(os.path.join(HERE, 'exact.parquet'))
