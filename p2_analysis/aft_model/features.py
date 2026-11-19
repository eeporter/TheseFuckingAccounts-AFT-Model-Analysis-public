"""Item-level flags and window statistics for the variables added in build v3 (2026-10-19): #4 burstiness, #5 comment-
thread length, #6b edit timing, #7 pace, #10 post-hopping, #14 self-repetition, #15 text length, #16 promotional links out
of place, #17 repeated interactions, #18 copied text, #8b replies on comments, #19 removal (extended model only).

Definitions where the plan gave only one line (defaults picked 2026-10-19, user may change them):
- #4  B = (sigma - mu) / (sigma + mu) of the gaps between the account's posts in the window; 0 with fewer than 3 posts.
      v4 (user, 2026-10-19) adds comment-thread burstiness: B over the gaps between the times the account joins a comment
      thread (its first comment in a thread, or its first after 7+ days away from it), so many comments inside one
      conversation count once; 0 with fewer than 3 thread entries in the window.
      v8 (user, 2026-10-22) replaces both with one measure over the merged stream of post times and comment-thread
      entries, using the finite-size burstiness of Kim & Jo (2016, Phys. Rev. E 94, 032311, Eq. 22):
      A_n = (sqrt(n+1) r - sqrt(n-1)) / ((sqrt(n+1) - 2) r + sqrt(n-1)), r = sigma/mu of the n gaps in the window
      (population sigma; the paper's n counts inter-event times, which equal events under its periodic boundary);
      -1 regular, 0 random (Poisson), 1 at the largest r possible for n gaps; 0 with fewer than 3 events.
- #5  turns of the longest alternating chain on one post (account, someone else, account, ...), counted in the account's
      turns; a chain is dated at its last account comment. A lone comment is 1; the account's own post counts as a turn
      only as the start of a chain. 0 with no comment in the window.
- #6b has edited (so far) and log(1 + median hours from item to edit) over the edits made so far.
- #7  median gap (hours) between consecutive posts / consecutive comments in the window; 168 (the window) if < 2 items.
- #10 median gap (hours) between consecutive comments in one subreddit on different posts, both in the window; 168 if none.
- #14 items (30+ characters) at least 90% similar (rapidfuzz ratio) to an earlier item by the same account.
- #15 mean characters per post (title + text) and per comment in the window.
- #16 one count: items with a porn, crypto, shop, YouTube/blog, messaging or news link outside that type's home groups.
- #17 partners (other accounts, either reply direction; AutoModerator and [deleted] excluded): with 2+ interactions in the
      window, and met in 2+ subreddits so far.
- #18 comments (30+ characters) at least 90% similar to (a) the text they replied to, or (b) an earlier comment by
      someone else in the account's own post thread.
- #19 posts removed by moderators or AutoModerator (removed_by_category moderator / automod_filtered) at post time, and
      comments that show as [removed] on Reddit today, at comment time.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

WIN = 7
MIN_CHARS = 30
SIM = 90
CAP_H = WIN * 24.0
HOME = {'porn': {'adult', 'profile'}, 'crypto': {'crypto'}, 'shop': {'sales'}, 'blog_channel': {'entertainment', 'profile'},
        'messaging': {'profile'}, 'news': {'news', 'politics'}}
NOT_PARTNER = {'automoderator', '[deleted]', 'none', 'nan', ''}
GONE = {'[removed]', '[deleted]'}


def norm(s):
    return s.fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()


def post_text(posts):
    body = posts.selftext.fillna('').where(~posts.selftext.isin(GONE), '')
    return (posts.title.fillna('') + ' ' + body).str.strip()


def comment_text(coms):
    return coms.body.fillna('').where(~coms.body.isin(GONE), '')


# --- item flags --------------------------------------------------------------------------------------------------

def self_repeats(items, block=2000):
    """items: outer_key, created_utc, text (index kept) -> bool Series: similar to an earlier item by the same account"""
    out = pd.Series(False, index=items.index)
    it = items.assign(n=norm(items.text))
    it = it[it.n.str.len() >= MIN_CHARS].sort_values(['outer_key', 'created_utc'])
    for _, g in it.groupby('outer_key', sort=False):
        txt = g.n.tolist()
        hit = np.zeros(len(txt), bool)
        for i0 in range(1, len(txt), block):
            i1 = min(i0 + block, len(txt))
            sc = process.cdist(txt[i0:i1], txt[:i1], scorer=fuzz.ratio, score_cutoff=SIM, dtype=np.uint8, workers=-1)
            sc[np.arange(i1 - i0)[:, None] <= (np.arange(i1)[None, :] - i0)] = 0      # earlier items only
            hit[i0:i1] = sc.max(axis=1) >= SIM
        out.loc[g.index[hit]] = True
    return out


def copied_parent(coms, parent_text):
    """(a) comment similar to the text it replied to; parent_text: parent_id, parent_author, parent_title, parent_text"""
    pt = parent_text.drop_duplicates('parent_id').set_index('parent_id')
    ptxt = (pt.parent_title.fillna('') + ' ' + pt.parent_text.fillna('').where(~pt.parent_text.isin(GONE), '')).str.strip()
    other = pt.parent_author.fillna('').str.lower()
    c = coms[['outer_key', 'parent_id', 'text']].copy()
    c['p'] = norm(c.parent_id.map(ptxt))
    c['n'] = norm(c.text)
    ok = (c.n.str.len() >= MIN_CHARS) & (c.p.str.len() >= MIN_CHARS) & (c.parent_id.map(other).fillna('') != c.outer_key)
    out = pd.Series(False, index=coms.index)
    if ok.any():
        sc = process.cpdist(c.n[ok].tolist(), c.p[ok].tolist(), scorer=fuzz.ratio, workers=-1)
        out.loc[c.index[ok]] = sc >= SIM
    return out


def copied_in_own_thread(coms, own_post_ids, others):
    """(b) comment in the account's own post thread similar to an earlier comment by someone else in that thread"""
    out = pd.Series(False, index=coms.index)
    c = coms[coms.link_id.str.removeprefix('t3_').isin(own_post_ids)].assign(n=lambda d: norm(d.text))
    c = c[c.n.str.len() >= MIN_CHARS]
    o = others.assign(n=norm(others.body.where(~others.body.isin(GONE), '')))
    o = o[o.n.str.len() >= MIN_CHARS]
    og = {k: g for k, g in o.groupby(['outer_key', 'link_id'])}
    for key, g in c.groupby(['outer_key', 'link_id']):
        if key not in og:
            continue
        r = og[key]
        sc = process.cdist(g.n.tolist(), r.n.tolist(), scorer=fuzz.ratio, score_cutoff=SIM, dtype=np.uint8, workers=-1)
        sc[g.created_utc.to_numpy()[:, None] <= r.created_utc.to_numpy()[None, :]] = 0
        out.loc[g.index[sc.max(axis=1) >= SIM]] = True
    return out


def thread_turns(coms, posts, others):
    """#5: turns of the alternating chain ending at each account comment (Series aligned with coms)"""
    other_parent = dict(zip(others.outer_key + '|' + others.id, others.parent_id))
    own = set(posts.outer_key + '|t3_' + posts.id)
    turns = {}
    out = np.ones(len(coms), int)
    order = np.argsort(coms.created_utc.to_numpy(), kind='stable')
    keys, ids, pars = coms.outer_key.to_numpy(), coms.id.to_numpy(), coms.parent_id.fillna('').to_numpy()
    for i in order:
        k, par = keys[i], pars[i]
        n = 1
        if par.startswith('t1_'):
            gp = other_parent.get(k + '|' + par[3:])            # parent written by someone else who replied to ...
            if gp is not None:
                if gp.startswith('t1_') and (k + '|' + gp[3:]) in turns:
                    n = turns[k + '|' + gp[3:]] + 1                 # ... an earlier account comment
                elif k + '|' + gp in own:
                    n = 2                                           # ... the account's own post
        turns[k + '|' + ids[i]] = n
        out[i] = n
    return pd.Series(out, index=coms.index)


def promo_out_of_place(links, items, kind):
    """#16 flags aligned with items (outer_key, id, group)"""
    L = links[links['kind'].eq(kind) & links.link_type.isin(list(HOME))][['outer_key', 'item_id', 'link_type']]
    g = items.set_index(['outer_key', 'id'])['group']
    grp = g.reindex(pd.MultiIndex.from_arrays([L.outer_key, L.item_id])).to_numpy()
    L = L.assign(out=[gr not in HOME[t] for t, gr in zip(L.link_type, grp)])
    f = L.groupby(['outer_key', 'item_id']).out.any()
    return f.reindex(pd.MultiIndex.from_arrays([items.outer_key, items.id]), fill_value=False).to_numpy()


# --- window statistics (window = (a - WIN, a] at each grid point a) ----------------------------------------------

def window_sum(times, vals, grid):
    o = np.argsort(times)
    s, v = np.asarray(times, float)[o], np.asarray(vals, float)[o]
    cs = np.r_[0, np.cumsum(v)]
    return cs[np.searchsorted(s, grid, 'right')] - cs[np.searchsorted(s, grid - WIN, 'right')]


def window_mean(times, vals, grid):
    n = window_sum(times, np.ones(len(times)), grid)
    return np.where(n > 0, window_sum(times, vals, grid) / np.maximum(n, 1), 0.0)


def window_max(times, vals, grid):
    o = np.argsort(times)
    s, v = np.asarray(times, float)[o], np.asarray(vals, float)[o]
    out = np.zeros(len(grid))
    if not len(s):
        return out
    lo, hi = np.searchsorted(s, grid - WIN, 'right'), np.searchsorted(s, grid, 'right')
    ok = hi > lo
    idx = np.empty(2 * ok.sum(), int)
    idx[0::2], idx[1::2] = lo[ok], hi[ok]
    red = np.maximum.reduceat(np.r_[v, 0], idx)                 # padded so hi == len(s) is a valid index
    out[ok] = red[0::2]
    return out


def burstiness(times, grid):
    s = np.sort(np.asarray(times, float))
    out = np.zeros(len(grid))
    if len(s) < 3:
        return out
    g = np.diff(s)
    c1, c2 = np.r_[0, np.cumsum(g)], np.r_[0, np.cumsum(g * g)]
    lo, hi = np.searchsorted(s, grid - WIN, 'right'), np.searchsorted(s, grid, 'right')
    n = hi - lo - 1                                             # gaps between items lo..hi-1 are g[lo:hi-1]
    ok = n >= 2
    a, b = lo[ok], hi[ok] - 1
    mu = (c1[b] - c1[a]) / n[ok]
    sd = np.sqrt(np.maximum((c2[b] - c2[a]) / n[ok] - mu ** 2, 0))
    out[ok] = np.where(sd + mu > 0, (sd - mu) / np.maximum(sd + mu, 1e-12), 0.0)
    return out


def thread_entries(c):
    """times the account joins a comment thread: first comment in a thread, or the first after a 7+ day gap in it"""
    if c.empty:
        return np.array([])
    d = c.sort_values(['thread', 't'])
    th, t = d.thread.to_numpy(), d.t.to_numpy()
    new = np.r_[True, (th[1:] != th[:-1]) | (np.diff(t) > WIN)]
    return np.sort(t[new])


def burstiness_kimjo(times, grid):
    """finite-size burstiness A_n (Kim & Jo 2016, Eq. 22) of the gaps in the window; 0 with fewer than 3 events"""
    s = np.sort(np.asarray(times, float))
    out = np.zeros(len(grid))
    if len(s) < 3:
        return out
    g = np.diff(s)
    c1, c2 = np.r_[0, np.cumsum(g)], np.r_[0, np.cumsum(g * g)]
    lo, hi = np.searchsorted(s, grid - WIN, 'right'), np.searchsorted(s, grid, 'right')
    n = hi - lo - 1
    ok = n >= 2
    a, b, k = lo[ok], hi[ok] - 1, n[ok].astype(float)
    mu = (c1[b] - c1[a]) / k
    sd = np.sqrt(np.maximum((c2[b] - c2[a]) / k - mu ** 2, 0))
    rr = np.where(mu > 0, sd / np.maximum(mu, 1e-12), 0.0)
    rr = np.minimum(rr, np.sqrt(k - 1))                         # float guard: r <= sqrt(n - 1) exactly
    num = np.sqrt(k + 1) * rr - np.sqrt(k - 1)
    den = (np.sqrt(k + 1) - 2) * rr + np.sqrt(k - 1)
    out[ok] = np.where(mu > 0, num / den, 0.0)                  # all events at one instant: no gap structure
    return out


def median_gap(t_prev, t_cur, grid):
    """log(1 + median hours) over gaps with both ends in the window; log(1 + 168) when there is none"""
    out = np.full(len(grid), np.log1p(CAP_H))
    if not len(t_cur):
        return out
    o = np.argsort(t_cur)
    tp, tc = np.asarray(t_prev, float)[o], np.asarray(t_cur, float)[o]
    gap = (tc - tp) * 24.0
    lo, hi = np.searchsorted(tc, grid - WIN, 'right'), np.searchsorted(tc, grid, 'right')
    cache = {}
    for j in np.nonzero(hi > lo)[0]:
        sel = np.arange(lo[j], hi[j])[tp[lo[j]:hi[j]] > grid[j] - WIN]   # gaps whose earlier end is also in the window
        if len(sel):
            k = sel.tobytes()
            if k not in cache:
                cache[k] = np.log1p(np.median(gap[sel]))
            out[j] = cache[k]
    return out


def consecutive(t):
    t = np.sort(np.asarray(t, float))
    return t[:-1], t[1:]


def hopping_pairs(c):
    """consecutive comments in one subreddit on different posts -> (t_prev, t_cur)"""
    tp, tc = [], []
    for _, s in c.sort_values('t').groupby('subreddit'):
        t, th = s.t.to_numpy(), s.thread.to_numpy()
        m = th[1:] != th[:-1]
        tp.append(t[:-1][m]); tc.append(t[1:][m])
    return (np.concatenate(tp), np.concatenate(tc)) if tp else (np.array([]), np.array([]))


def edit_timing(edit_t, item_t, grid):
    """has edited so far, log(1 + median hours to edit) over edits so far"""
    ok = ~np.isnan(edit_t)
    e, d = np.asarray(edit_t)[ok], (np.asarray(edit_t)[ok] - np.asarray(item_t)[ok]) * 24.0
    o = np.argsort(e); e, d = e[o], np.maximum(d[o], 0)
    k = np.searchsorted(e, grid, 'right')
    med = np.r_[0.0, [np.log1p(np.median(d[:i + 1])) for i in range(len(d))]]
    return (k > 0).astype(float), med[k]


def repeat_partners(t, partner, grid):
    """partners with 2+ interactions in the window"""
    if not len(t):
        return np.zeros(len(grid))
    df = pd.DataFrame({'t': t, 'p': partner}).sort_values(['p', 't'])
    tt, pp = df.t.to_numpy(), df.p.to_numpy()
    same = pp[1:] == pp[:-1]
    st, en, who = tt[1:][same], tt[:-1][same] + WIN, pp[1:][same]   # both in window while t_k <= a < t_(k-1) + WIN
    keep = st < en
    st, en, who = st[keep], en[keep], who[keep]
    if not len(st):
        return np.zeros(len(grid))
    iv = pd.DataFrame({'s': st, 'e': en, 'p': who})            # merge overlaps within a partner (starts and ends rise)
    prev_end = iv.groupby('p').e.transform(lambda e: e.shift().cummax())   # within the partner only
    iv['blk'] = ((iv.p != iv.p.shift()) | (iv.s > prev_end)).cumsum()
    m = iv.groupby('blk').agg(s=('s', 'min'), e=('e', 'max'))
    return (np.searchsorted(np.sort(m.s.to_numpy()), grid, 'right') - np.searchsorted(np.sort(m.e.to_numpy()), grid, 'right')).astype(float)


def cross_sub_partners(t, partner, sub, grid):
    """partners met in 2+ subreddits so far"""
    if not len(t):
        return np.zeros(len(grid))
    df = pd.DataFrame({'t': t, 'p': partner, 's': sub}).sort_values('t')
    first = df.drop_duplicates(['p', 's'])                       # first meeting in each subreddit
    second = first.groupby('p').t.nth(1)                         # time the second subreddit is reached
    return np.searchsorted(np.sort(second.to_numpy()), grid, 'right').astype(float)
