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

What this module does: a library imported by build.py (as F); it reads and writes no files itself. Two kinds of function:
- item flags (self_repeats ... promo_out_of_place): one value per post or comment, computed once over all accounts;
  build.py stores them as item columns and then counts flagged items in the window.
- window statistics (window_sum ... cross_sub_partners): one value per grid point of one account. Times are days since
  the account's first archived action (float); grid holds the episode start times a (days, ascending). "Window" means
  the items with time in (a - WIN, a]; "so far" means all items with time <= a. Gaps are reported in hours.
Not run on its own; run build.py.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

WIN = 7                 # window length, days
MIN_CHARS = 30          # shortest normalised text compared for similarity (#14, #18)
SIM = 90                # rapidfuzz ratio (0-100) counted as similar
CAP_H = WIN * 24.0      # 168 h: gap reported when the window holds no gap (#7, #10)
# #16: link type -> subreddit groups (subreddit_groups_v9) where that link is at home; anywhere else it is out of place
HOME = {'porn': {'adult', 'profile'}, 'crypto': {'crypto'}, 'shop': {'sales'}, 'blog_channel': {'entertainment', 'profile'},
        'messaging': {'profile'}, 'news': {'news', 'politics'}}
NOT_PARTNER = {'automoderator', '[deleted]', 'none', 'nan', ''}   # lowercased author values that are not a partner (#17)
GONE = {'[removed]', '[deleted]'}                                 # placeholder bodies, treated as empty text


def norm(s):
    """Text Series -> lowercased, whitespace collapsed and stripped (NaN -> ''), for similarity comparisons."""
    return s.fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()


def post_text(posts):
    """Post text = title + ' ' + selftext, with a [removed]/[deleted] selftext treated as empty. Series aligned with posts."""
    body = posts.selftext.fillna('').where(~posts.selftext.isin(GONE), '')
    return (posts.title.fillna('') + ' ' + body).str.strip()


def comment_text(coms):
    """Comment body with [removed]/[deleted] treated as empty. Series aligned with coms."""
    return coms.body.fillna('').where(~coms.body.isin(GONE), '')


# --- item flags --------------------------------------------------------------------------------------------------

def self_repeats(items, block=2000):
    """items: outer_key, created_utc, text (index kept) -> bool Series: similar to an earlier item by the same account

    #14. Posts and comments together; only normalised texts of MIN_CHARS+ are compared, others stay False. "Earlier" is
    the order after sorting by created_utc, so of two same-second items only the second can be flagged. Rows are
    scored in blocks of `block` against all items up to the block end, to bound the size of the score matrix.
    """
    out = pd.Series(False, index=items.index)
    it = items.assign(n=norm(items.text))
    it = it[it.n.str.len() >= MIN_CHARS].sort_values(['outer_key', 'created_utc'])
    for _, g in it.groupby('outer_key', sort=False):
        txt = g.n.tolist()
        hit = np.zeros(len(txt), bool)
        for i0 in range(1, len(txt), block):                    # item 0 has nothing earlier
            i1 = min(i0 + block, len(txt))
            sc = process.cdist(txt[i0:i1], txt[:i1], scorer=fuzz.ratio, score_cutoff=SIM, dtype=np.uint8, workers=-1)
            sc[np.arange(i1 - i0)[:, None] <= (np.arange(i1)[None, :] - i0)] = 0      # earlier items only
            hit[i0:i1] = sc.max(axis=1) >= SIM
        out.loc[g.index[hit]] = True
    return out


def copied_parent(coms, parent_text):
    """(a) comment similar to the text it replied to; parent_text: parent_id, parent_author, parent_title, parent_text

    #18a. coms: outer_key, parent_id, text. The parent's text is its title + text ([removed]/[deleted] text empty).
    Both texts must be MIN_CHARS+ after normalising, and replies to the account's own items are skipped.
    Returns a bool Series aligned with coms.
    """
    pt = parent_text.drop_duplicates('parent_id').set_index('parent_id')
    ptxt = (pt.parent_title.fillna('') + ' ' + pt.parent_text.fillna('').where(~pt.parent_text.isin(GONE), '')).str.strip()
    other = pt.parent_author.fillna('').str.lower()
    c = coms[['outer_key', 'parent_id', 'text']].copy()
    c['p'] = norm(c.parent_id.map(ptxt))
    c['n'] = norm(c.text)
    ok = (c.n.str.len() >= MIN_CHARS) & (c.p.str.len() >= MIN_CHARS) & (c.parent_id.map(other).fillna('') != c.outer_key)
    out = pd.Series(False, index=coms.index)
    if ok.any():
        sc = process.cpdist(c.n[ok].tolist(), c.p[ok].tolist(), scorer=fuzz.ratio, workers=-1)   # pairwise: comment i vs its parent i
        out.loc[c.index[ok]] = sc >= SIM
    return out


def copied_in_own_thread(coms, own_post_ids, others):
    """(b) comment in the account's own post thread similar to an earlier comment by someone else in that thread

    #18b. coms: outer_key, link_id, created_utc, text; own_post_ids: set of bare post ids (no t3_ prefix);
    others: other people's comments with outer_key, link_id, created_utc, body. Comments are matched within the same
    (outer_key, link_id); "earlier" is strictly earlier created_utc. Returns a bool Series aligned with coms.
    """
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
        sc[g.created_utc.to_numpy()[:, None] <= r.created_utc.to_numpy()[None, :]] = 0   # keep others' comments made before
        out.loc[g.index[sc.max(axis=1) >= SIM]] = True
    return out


def thread_turns(coms, posts, others):
    """#5: turns of the alternating chain ending at each account comment (Series aligned with coms)

    coms: outer_key, id (bare), parent_id (t1_/t3_ prefixed), created_utc; posts: outer_key, id (bare);
    others: other people's comments with outer_key, id (bare), parent_id. A comment that replies to someone else's
    comment, which itself replied to an account comment with n turns, gets n + 1; if that other comment replied to the
    account's own post, it gets 2. Every other comment (top-level, reply to itself, or no known chain) gets 1.
    """
    other_parent = dict(zip(others.outer_key + '|' + others.id, others.parent_id))   # other's comment -> what it replied to
    own = set(posts.outer_key + '|t3_' + posts.id)
    turns = {}                                                  # 'outer_key|comment id' -> turns, filled in time order
    out = np.ones(len(coms), int)
    order = np.argsort(coms.created_utc.to_numpy(), kind='stable')   # chain predecessors are always processed first
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
    """#16 flags aligned with items (outer_key, id, group)

    links: the link table (links.build) with kind ('post'/'comment'), outer_key, item_id, link_type. An item is flagged
    when any of its links has a type in HOME and the item's subreddit group is not one of that type's home groups.
    Returns a bool numpy array, one value per row of items.
    """
    L = links[links['kind'].eq(kind) & links.link_type.isin(list(HOME))][['outer_key', 'item_id', 'link_type']]
    g = items.set_index(['outer_key', 'id'])['group']
    grp = g.reindex(pd.MultiIndex.from_arrays([L.outer_key, L.item_id])).to_numpy()
    L = L.assign(out=[gr not in HOME[t] for t, gr in zip(L.link_type, grp)])
    f = L.groupby(['outer_key', 'item_id']).out.any()
    return f.reindex(pd.MultiIndex.from_arrays([items.outer_key, items.id]), fill_value=False).to_numpy()


# --- window statistics (window = (a - WIN, a] at each grid point a) ----------------------------------------------
# On sorted times s, searchsorted(s, a, 'right') = number of items with time <= a, so the items in (a - WIN, a] are
# s[searchsorted(s, a - WIN, 'right') : searchsorted(s, a, 'right')]: 'right' at both ends puts an item at exactly a
# inside the window and one at exactly a - WIN (the moment it leaves) outside.

def window_sum(times, vals, grid):
    """Sum of vals over items with time in (a - WIN, a], one value per grid point. times in days."""
    o = np.argsort(times)
    s, v = np.asarray(times, float)[o], np.asarray(vals, float)[o]
    cs = np.r_[0, np.cumsum(v)]                                 # cs[i] = sum of the first i values
    return cs[np.searchsorted(s, grid, 'right')] - cs[np.searchsorted(s, grid - WIN, 'right')]


def window_mean(times, vals, grid):
    """Mean of vals over items in the window, one value per grid point; 0 when the window is empty."""
    n = window_sum(times, np.ones(len(times)), grid)
    return np.where(n > 0, window_sum(times, vals, grid) / np.maximum(n, 1), 0.0)


def window_max(times, vals, grid):
    """Max of vals over items in the window, one value per grid point; 0 when the window is empty."""
    o = np.argsort(times)
    s, v = np.asarray(times, float)[o], np.asarray(vals, float)[o]
    out = np.zeros(len(grid))
    if not len(s):
        return out
    lo, hi = np.searchsorted(s, grid - WIN, 'right'), np.searchsorted(s, grid, 'right')
    ok = hi > lo
    idx = np.empty(2 * ok.sum(), int)
    idx[0::2], idx[1::2] = lo[ok], hi[ok]                       # reduceat over v[lo:hi]; the odd slots are discarded
    red = np.maximum.reduceat(np.r_[v, 0], idx)                 # padded so hi == len(s) is a valid index
    out[ok] = red[0::2]
    return out


def burstiness(times, grid):
    """#4 (v3/v4): B = (sigma - mu) / (sigma + mu) of the gaps between consecutive items in the window.

    times: event times in days (any order). Only gaps with both ends in the window count; population sigma.
    0 with fewer than 2 gaps (3 events) in the window, or when sigma + mu = 0. One value per grid point.
    """
    s = np.sort(np.asarray(times, float))
    out = np.zeros(len(grid))
    if len(s) < 3:
        return out
    g = np.diff(s)
    c1, c2 = np.r_[0, np.cumsum(g)], np.r_[0, np.cumsum(g * g)]   # prefix sums of gaps and squared gaps
    lo, hi = np.searchsorted(s, grid - WIN, 'right'), np.searchsorted(s, grid, 'right')
    n = hi - lo - 1                                             # gaps between items lo..hi-1 are g[lo:hi-1]
    ok = n >= 2
    a, b = lo[ok], hi[ok] - 1
    mu = (c1[b] - c1[a]) / n[ok]                                # mean gap = sum(g[a:b]) / n
    sd = np.sqrt(np.maximum((c2[b] - c2[a]) / n[ok] - mu ** 2, 0))   # sqrt(E[g^2] - mu^2); max(., 0) for rounding
    out[ok] = np.where(sd + mu > 0, (sd - mu) / np.maximum(sd + mu, 1e-12), 0.0)
    return out


def thread_entries(c):
    """times the account joins a comment thread: first comment in a thread, or the first after a 7+ day gap in it

    c: the account's comments with thread (post id) and t (days). The gap test is strictly more than WIN days.
    Returns a sorted array of entry times (days).
    """
    if c.empty:
        return np.array([])
    d = c.sort_values(['thread', 't'])
    th, t = d.thread.to_numpy(), d.t.to_numpy()
    new = np.r_[True, (th[1:] != th[:-1]) | (np.diff(t) > WIN)]
    return np.sort(t[new])


def burstiness_kimjo(times, grid):
    """finite-size burstiness A_n (Kim & Jo 2016, Eq. 22) of the gaps in the window; 0 with fewer than 3 events

    times: event times in days (build passes post times + comment-thread entries). n = number of gaps with both ends
    in the window, r = sigma / mu over them (population sigma), computed with prefix sums as in burstiness().
    0 also when every gap is 0. One value per grid point.
    """
    s = np.sort(np.asarray(times, float))
    out = np.zeros(len(grid))
    if len(s) < 3:
        return out
    g = np.diff(s)
    c1, c2 = np.r_[0, np.cumsum(g)], np.r_[0, np.cumsum(g * g)]
    lo, hi = np.searchsorted(s, grid - WIN, 'right'), np.searchsorted(s, grid, 'right')
    n = hi - lo - 1                                             # gaps in the window, as in burstiness()
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
    """log(1 + median hours) over gaps with both ends in the window; log(1 + 168) when there is none

    #7 / #10. t_prev, t_cur: paired gap ends in days (from consecutive() or hopping_pairs()). A gap is in the window
    when t_cur is in (a - WIN, a] and t_prev > a - WIN. Returns log scale, one value per grid point.
    """
    out = np.full(len(grid), np.log1p(CAP_H))                   # filler: no gap in the window = the whole window, 168 h
    if not len(t_cur):
        return out
    o = np.argsort(t_cur)
    tp, tc = np.asarray(t_prev, float)[o], np.asarray(t_cur, float)[o]
    gap = (tc - tp) * 24.0                                      # days -> hours
    lo, hi = np.searchsorted(tc, grid - WIN, 'right'), np.searchsorted(tc, grid, 'right')
    cache = {}                                                  # same set of gaps -> same median; many grid points share one
    for j in np.nonzero(hi > lo)[0]:
        sel = np.arange(lo[j], hi[j])[tp[lo[j]:hi[j]] > grid[j] - WIN]   # gaps whose earlier end is also in the window
        if len(sel):
            k = sel.tobytes()
            if k not in cache:
                cache[k] = np.log1p(np.median(gap[sel]))
            out[j] = cache[k]
    return out


def consecutive(t):
    """Times (days) -> (t_prev, t_cur): the gap ends between consecutive items in time order."""
    t = np.sort(np.asarray(t, float))
    return t[:-1], t[1:]


def hopping_pairs(c):
    """consecutive comments in one subreddit on different posts -> (t_prev, t_cur)

    #10. c: the account's comments with t (days), subreddit, thread. Within each subreddit, comments in time order; a
    pair is kept when the two consecutive comments are on different posts. Returns two arrays in days.
    """
    tp, tc = [], []
    for _, s in c.sort_values('t').groupby('subreddit'):
        t, th = s.t.to_numpy(), s.thread.to_numpy()
        m = th[1:] != th[:-1]
        tp.append(t[:-1][m]); tc.append(t[1:][m])
    return (np.concatenate(tp), np.concatenate(tc)) if tp else (np.array([]), np.array([]))


def edit_timing(edit_t, item_t, grid):
    """has edited so far, log(1 + median hours to edit) over edits so far

    #6b. edit_t: edit time per item in days (NaN = not edited, or edit outside the study period); item_t: the item's
    creation time in days. Both outputs are "so far" (edits with time <= a), one value per grid point; the median is
    0 before the first edit. A negative delay is set to 0.
    """
    ok = ~np.isnan(edit_t)
    e, d = np.asarray(edit_t)[ok], (np.asarray(edit_t)[ok] - np.asarray(item_t)[ok]) * 24.0
    o = np.argsort(e); e, d = e[o], np.maximum(d[o], 0)        # edits in the order they were made
    k = np.searchsorted(e, grid, 'right')                       # edits made by a
    med = np.r_[0.0, [np.log1p(np.median(d[:i + 1])) for i in range(len(d))]]   # med[k] = median over the first k edits
    return (k > 0).astype(float), med[k]


def repeat_partners(t, partner, grid):
    """partners with 2+ interactions in the window

    #17. t: interaction times in days; partner: the other account (lowercased), same length. Returns the number of
    distinct partners with at least two interactions in (a - WIN, a], one float per grid point.
    """
    if not len(t):
        return np.zeros(len(grid))
    df = pd.DataFrame({'t': t, 'p': partner}).sort_values(['p', 't'])
    tt, pp = df.t.to_numpy(), df.p.to_numpy()
    same = pp[1:] == pp[:-1]                                    # consecutive interactions with the same partner
    st, en, who = tt[1:][same], tt[:-1][same] + WIN, pp[1:][same]   # both in window while t_k <= a < t_(k-1) + WIN
    keep = st < en
    st, en, who = st[keep], en[keep], who[keep]
    if not len(st):
        return np.zeros(len(grid))
    # a partner counts while any of its intervals [s, e) covers a; merging each partner's overlapping intervals into
    # blocks means a partner is counted once, so the count is (blocks started by a) - (blocks ended by a)
    iv = pd.DataFrame({'s': st, 'e': en, 'p': who})            # merge overlaps within a partner (starts and ends rise)
    prev_end = iv.groupby('p').e.transform(lambda e: e.shift().cummax())   # within the partner only
    iv['blk'] = ((iv.p != iv.p.shift()) | (iv.s > prev_end)).cumsum()       # new block: new partner or a gap
    m = iv.groupby('blk').agg(s=('s', 'min'), e=('e', 'max'))
    return (np.searchsorted(np.sort(m.s.to_numpy()), grid, 'right') - np.searchsorted(np.sort(m.e.to_numpy()), grid, 'right')).astype(float)


def cross_sub_partners(t, partner, sub, grid):
    """partners met in 2+ subreddits so far

    #17. t: interaction times in days; partner: lowercased other account; sub: subreddit of the interaction.
    Returns, per grid point, the number of partners whose second distinct subreddit was reached at time <= a (float).
    """
    if not len(t):
        return np.zeros(len(grid))
    df = pd.DataFrame({'t': t, 'p': partner, 's': sub}).sort_values('t')
    first = df.drop_duplicates(['p', 's'])                       # first meeting in each subreddit
    second = first.groupby('p').t.nth(1)                         # time the second subreddit is reached
    return np.searchsorted(np.sort(second.to_numpy()), grid, 'right').astype(float)
