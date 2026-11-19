"""Exact-time covariates for the AFT pilot with the rules decided so far (rules.py).

Same variable subset as the 9 Oct test fit (aft_pilot_test/build_exact.py); the remaining variables are added here as
they are built. Changes from the test build:
- capped accounts are dropped (not censored): v1 dropped the 7 whose archive ends before the accusation; v2 (user,
  2026-10-16) drops all 11 capped accounts (rules.capped_accounts)
- visual media / external links follow the render rule (rules.link_renders), using the link table from Link_domains_v1
- flooding needs 10+ threads in the busiest subreddit (rules.flooding)
- v3 (2026-10-19) adds the variables that were not yet built (features.py: #4, #5, #6b, #7, #10, #14-#18, replies on
  comments, and #19 removal for the extended model); the v2 columns are unchanged, only split at the new change points
- v4 (2026-10-19, user) adds comment-thread burstiness (features.thread_entries); everything else as v3
- v5 (2026-10-22, user) adds the window counts behind the active-week terms: posts, comments and thread entries in
  the last 7 days (n_posts_7d, n_comments_7d, n_entries_7d); everything else as v4
- v6 (2026-10-22, user) adds return after silence: at each post or comment, days since the account's previous post or
  comment (0 for its first item); return_gap = the largest such gap among items in the last 7 days, 0 when inactive
- v7 (2026-10-22, user) splits post length: chars_title = mean title characters over posts in the window;
  chars_body = mean selftext characters ([removed]/[deleted] empty) over self-posts in the window; n_selfposts_7d
- v8 (2026-10-22, user) adds one burstiness over posts and comment-thread entries together (features.burstiness_kimjo,
  Kim & Jo 2016 finite-size measure) and its event count n_burst_7d; the two v4 burstiness columns stay for comparison
- v9 (2026-10-27, user) adds shared_text: posts and comments in the window whose text (60+ characters, normalised)
  matches an earlier or same-second item by a different accused account (../shared_text_v1.py, all stored histories)
Covariates change at the moment of each action and when it leaves the 7-day window. Writes aft_model_output/.

`python build.py --to-cutoff` continues every account past its accusation up to the sample cutoff (2022-11-30, pool
rule 5) with its archived behaviour, for the right-truncation term; rows after the accusation have after = 1. Rows
before the accusation are the same as in exact_<version>. After the accusation, subreddits outside the v9 groups count as
'other', and replies are only those on posts made before the accusation (the only threads collected).
"""
import os, sys, glob, re, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rules
import links as linkbuild
import features as F

P2 = os.path.dirname(HERE)
S = os.path.join(P2, 'sample_998_v4_excl_output')
OUT = os.path.join(P2, 'aft_model_output')
DAY = 86400.0
WIN = 7
CUTOFF = rules.CUTOFF
TO_CUTOFF = '--to-cutoff' in sys.argv
GROUPS = ['profile', 'adult', 'crypto', 'politics', 'karma_farming', 'general', 'news', 'sales', 'location', 'entertainment', 'other']


def save_new(df, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        raise FileExistsError(f'exists, not overwritten: {path}')
    df.to_parquet(path, engine='pyarrow', compression='zstd')
    print(f'wrote {path}  {df.shape}', flush=True)


t0 = time.time()
a_full = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
posts_all = pd.read_parquet(os.path.join(S, 'sample_posts.parquet'), columns=[
    'id', 'outer_key', 'created_utc', 'subreddit', 'title', 'selftext', 'url', 'is_self', 'over_18', 'score', 'edited',
    'removed_by_category'])
coms_all = pd.read_parquet(os.path.join(S, 'sample_comments.parquet'), columns=[
    'id', 'outer_key', 'created_utc', 'subreddit', 'link_id', 'parent_id', 'body', 'score', 'edited'])
VERSION = 'v9'                              # v1: 7 capped dropped; v2: 9 Oct only; v3: + planned; v4: + thread burstiness;
                                            # v5: + week counts; v6: + return gap; v7: + title/body length;
                                            # v8: + Kim & Jo burstiness
DROPPED = rules.capped_accounts(a_full, posts_all, coms_all)
print('dropped (capped archive):', DROPPED, flush=True)
acc = a_full.drop(DROPPED)[['username', 'first_action_utc', 'first_mention_adj_utc', 'days_to_accused']]
LIMIT = int(os.environ.get('BUILD_LIMIT', 0))         # quick test on the first LIMIT accounts -> test_*.parquet
if LIMIT:
    acc = acc.head(LIMIT)
groups = pd.read_parquet(os.path.join(P2, 'subreddit_review_output', 'subreddit_groups_v9.parquet')).set_index('subreddit')['group']

posts = posts_all.drop_duplicates(['outer_key', 'id'])
coms = coms_all.drop_duplicates(['outer_key', 'id'])
posts, coms = posts[posts.outer_key.isin(acc.index)].copy(), coms[coms.outer_key.isin(acc.index)].copy()
for d in (posts, coms):
    d['acc_utc'] = d['outer_key'].map(acc['first_mention_adj_utc'])
    d['t0'] = d['outer_key'].map(acc['first_action_utc'])
    d['until'] = CUTOFF if TO_CUTOFF else d['acc_utc']       # items (and edits) used
posts = posts[posts.created_utc < posts.until].copy()
coms = coms[coms.created_utc < coms.until].copy()

links = linkbuild.build(posts, coms)      # same as link_domain_output/links_pre_accusation_v1 on pre-accusation items
posts['visual'], posts['external'] = rules.item_link_flags(links, posts, 'post', 'selftext')
coms['visual'], coms['external'] = rules.item_link_flags(links, coms, 'comment', 'body')

xp = pd.read_parquet(os.path.join(P2, 'crosspost_output', 'crosspost_flags_998.parquet'), columns=['id', 'outer_key', 'native_crosspost'])
XPOST = re.compile(r'(?<![a-z])(x-?\s?post(?:ed)?|cross-?\s?post(?:ed)?)(?![a-z])', re.I)   # URLs stripped first
posts['xtext'] = (posts.title.fillna('') + ' ' + posts.selftext.fillna('').str[:500]).str.replace(r'https?://\S+', ' ', regex=True)
posts = posts.merge(xp, on=['id', 'outer_key'], how='left')
posts['crosspost'] = posts.native_crosspost.fillna(False).astype(bool) | posts.xtext.str.contains(XPOST)

posts['group'] = posts.subreddit.map(groups)
coms['group'] = coms.subreddit.map(groups)
for d in (posts, coms):
    assert d.group[d.created_utc < d.acc_utc].notna().all(), 'subreddit missing from v9 groups'
    d['group'] = d['group'].fillna('other')                 # only after the accusation (--to-cutoff)
posts['nsfw_outside'] = posts.over_18.astype(str).eq('True') & ~posts.group.isin(['adult', 'profile'])
posts['score'] = pd.to_numeric(posts.score); coms['score'] = pd.to_numeric(coms.score)
posts['down'] = posts.score.le(0); coms['down'] = coms.score.lt(0)
posts['popular'] = posts.score.ge(100); coms['popular'] = coms.score.ge(100)
posts['thread'] = posts['id']
coms['thread'] = coms.link_id.str.removeprefix('t3_')
for d in (posts, coms):
    d['t'] = (d.created_utc - d.t0) / DAY
    e = pd.to_numeric(d.edited.replace('False', np.nan), errors='coerce')
    d['edit_t'] = np.where(e.notna() & (e < d['until']) & (e >= d.t0), (e - d.t0) / DAY, np.nan)

# dated replies received on own posts
rep = pd.concat(pd.read_parquet(f, columns=['author', 'created_utc', 'source_id', 'outer_key'])
                for f in sorted(glob.glob(os.path.join(P2, 'replies_output', 'posts_parts', 'replies_*.parquet'))))
rep = rep[(rep.author.str.lower() != rep.outer_key) & rep.outer_key.isin(acc.index)].copy()
rep['created_utc'] = pd.to_numeric(rep.created_utc)
rep = rep[rep.created_utc < (CUTOFF if TO_CUTOFF else rep.outer_key.map(acc['first_mention_adj_utc']))]
rep['t'] = (rep.created_utc - rep.outer_key.map(acc['first_action_utc'])) / DAY

# --- v3 variables (features.py) ---
def read_parts(pattern, **kw):
    return pd.concat([d for d in (pd.read_parquet(f, **kw) for f in sorted(glob.glob(os.path.join(P2, pattern)))) if len(d)])


# everyone else's comments in the account's post threads and direct replies to its comments
others = pd.concat([read_parts('replies_output/posts_parts/replies_*.parquet'), read_parts('replies_output/comments_parts/replies_*.parquet')])
others = others[others.id.notna() & others.outer_key.isin(acc.index)].drop_duplicates(['outer_key', 'id'])
others = others[others.author.fillna('').str.lower() != others.outer_key].copy()
others['created_utc'] = pd.to_numeric(others.created_utc)
others = others[others.created_utc < (CUTOFF if TO_CUTOFF else others.outer_key.map(acc['first_mention_adj_utc']))]
others['t'] = (others.created_utc - others.outer_key.map(acc['first_action_utc'])) / DAY
own_items = set('t3_' + posts.id) | set('t1_' + coms.id)
others['to_account'] = others.parent_id.isin(own_items)     # a direct reply to one of the account's items
rc = others[others.parent_id.fillna('').str.startswith('t1_') & others.to_account]   # replies on comments (#8b)

posts['text'] = F.post_text(posts); coms['text'] = F.comment_text(coms)
posts['chars'] = posts.text.str.len(); coms['chars'] = coms.text.str.len()
posts['self'] = posts.is_self.astype(str).eq('True')
posts['chars_title'] = posts.title.fillna('').str.len()
posts['chars_body'] = posts.selftext.fillna('').where(~posts.selftext.isin(F.GONE), '').str.len()
items = pd.concat([posts[['outer_key', 'created_utc', 'text']], coms[['outer_key', 'created_utc', 'text']]], keys=['p', 'c'])
srep = F.self_repeats(items)
posts['self_repeat'], coms['self_repeat'] = srep.loc['p'].to_numpy(), srep.loc['c'].to_numpy()
print(f'self-repetition {time.time() - t0:.0f}s', flush=True)
ptext = read_parts('parent_text_output/parts/*.parquet')
coms['copied'] = (F.copied_parent(coms, ptext) | F.copied_in_own_thread(coms, set(posts.id), others)).to_numpy()
coms['turns'] = F.thread_turns(coms, posts, others).to_numpy()
posts['promo_out'] = F.promo_out_of_place(links, posts, 'post')
coms['promo_out'] = F.promo_out_of_place(links, coms, 'comment')
posts['removed_mod'] = posts.removed_by_category.isin(['moderator', 'automod_filtered'])
crem = read_parts('comment_removal_output/parts/*.parquet')
crem = crem[crem.body_status.eq('removed')]
coms['removed_now'] = pd.MultiIndex.from_arrays([coms.outer_key, coms.id]).isin(pd.MultiIndex.from_arrays([crem.outer_key, crem.id]))
# interactions for #17: the author of what the account replied to, and everyone who replied to the account
pauth = read_parts('parent_authors_output/parts/*.parquet').drop_duplicates('parent_id').set_index('parent_id').parent_author
inter = pd.concat([
    pd.DataFrame({'outer_key': coms.outer_key, 't': coms.t, 'partner': coms.parent_id.map(pauth), 'sub': coms.subreddit}),
    pd.DataFrame({'outer_key': others.outer_key[others.to_account], 't': others.t[others.to_account],
                  'partner': others.author[others.to_account], 'sub': others.subreddit[others.to_account]})])
inter['partner'] = inter.partner.fillna('').astype(str).str.lower()
inter = inter[~inter.partner.isin(F.NOT_PARTNER) & (inter.partner != inter.outer_key)]
shared = pd.read_parquet(os.path.join(P2, 'shared_text_output', 'shared_text_flags_v1.parquet'), columns=['kind', 'outer_key', 'id', 'shared'])
for d, kind in ((posts, 'post'), (coms, 'comment')):
    s = shared[shared.kind.eq(kind) & shared.shared]
    d['shared'] = pd.MultiIndex.from_arrays([d.outer_key, d.id]).isin(pd.MultiIndex.from_arrays([s.outer_key, s.id]))
print(f'v3 item flags {time.time() - t0:.0f}s', flush=True)

# TFA posts in the previous 30 days (calendar), 2014-2022 corpus
tfa = np.sort(pd.read_csv('/data/Emily_data/TheseFuckingAccounts_submissions.csv', usecols=['created_utc'])
              .created_utc.pipe(pd.to_numeric, errors='coerce').dropna().to_numpy())
print(f'loaded {time.time() - t0:.0f}s', flush=True)


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


rows = []
pg, cg, rg = posts.groupby('outer_key'), coms.groupby('outer_key'), rep.groupby('outer_key')
rcg, ig = rc.groupby('outer_key'), inter.groupby('outer_key')
for k, a in acc.iterrows():
    T = a.days_to_accused
    end = (CUTOFF - a.first_action_utc) / DAY if TO_CUTOFF else T
    p = pg.get_group(k) if k in pg.groups else posts.iloc[:0]
    c = cg.get_group(k) if k in cg.groups else coms.iloc[:0]
    r = rg.get_group(k) if k in rg.groups else rep.iloc[:0]
    q = rcg.get_group(k) if k in rcg.groups else rc.iloc[:0]
    it = ig.get_group(k) if k in ig.groups else inter.iloc[:0]
    times = np.r_[p.t, c.t, p.edit_t.dropna(), c.edit_t.dropna(), r.t, q.t]
    pts = np.r_[0.0, T, times, times + WIN, np.arange(1, np.ceil(end))]
    act = pd.concat([p[['thread', 't', 'group', 'subreddit']], c[['thread', 't', 'group', 'subreddit']]])
    if len(act):                                  # thread-block ends also change the thread counts
        pts = np.r_[pts, act.groupby('thread').t.max().to_numpy() + WIN]
    grid = np.unique(pts[(pts >= 0) & (pts < end)])
    after = grid >= T
    f = {'id': np.repeat(k, len(grid)), 'start': grid, 'stop': np.r_[grid[1:], end],
         'event': (np.r_[grid[1:], end] == T).astype(float), 'after': after.astype(float)}
    for gname in GROUPS:
        f[f'thr_{gname}'] = thread_active(act[act.group == gname], grid)
    tot = np.zeros(len(grid)); mx = np.zeros(len(grid))
    for _, s in act.groupby('subreddit'):
        w = thread_active(s, grid); tot += w; mx = np.maximum(mx, w)
    f['flooding'] = rules.flooding(mx, tot)
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
    # v3 variables
    f['burstiness'] = F.burstiness(p.t, grid)
    f['burstiness_threads'] = F.burstiness(F.thread_entries(c), grid)
    f['n_posts_7d'] = window(p.t, grid)
    f['n_comments_7d'] = window(c.t, grid)
    f['n_entries_7d'] = window(F.thread_entries(c), grid)
    at = np.sort(np.r_[p.t, c.t])
    f['return_gap'] = F.window_max(at, np.r_[0.0, np.diff(at)], grid)
    f['chars_title'] = F.window_mean(p.t, p.chars_title, grid)
    f['chars_body'] = F.window_mean(p.t[p.self], p.chars_body[p.self], grid)
    f['n_selfposts_7d'] = window(p.t[p.self], grid)
    ev = np.r_[p.t, F.thread_entries(c)]
    f['burstiness_all'] = F.burstiness_kimjo(ev, grid)
    f['n_burst_7d'] = window(ev, grid)
    f['shared_text'] = window(np.r_[p.t[p.shared], c.t[c.shared]], grid)
    f['thread_turns'] = F.window_max(c.t, c.turns, grid)
    f['has_edited'], f['edit_delay'] = F.edit_timing(np.r_[p.edit_t, c.edit_t], np.r_[p.t, c.t], grid)
    f['pace_posts'] = F.median_gap(*F.consecutive(p.t), grid)
    f['pace_comments'] = F.median_gap(*F.consecutive(c.t), grid)
    f['hopping'] = F.median_gap(*F.hopping_pairs(c), grid)
    f['self_repeats'] = window(np.r_[p.t[p.self_repeat], c.t[c.self_repeat]], grid)
    f['chars_posts'] = F.window_mean(p.t, p.chars, grid)
    f['chars_comments'] = F.window_mean(c.t, c.chars, grid)
    f['promo_out_of_place'] = window(np.r_[p.t[p.promo_out], c.t[c.promo_out]], grid)
    f['repeat_partners'] = F.repeat_partners(it.t.to_numpy(), it.partner.to_numpy(), grid)
    f['cross_sub_partners'] = F.cross_sub_partners(it.t.to_numpy(), it.partner.to_numpy(), it['sub'].to_numpy(), grid)
    f['copied'] = window(c.t[c.copied], grid)
    f['replies_on_comments'] = window(q.t, grid)
    f['removed_posts'] = window(p.t[p.removed_mod], grid)
    f['removed_comments'] = window(c.t[c.removed_now], grid)
    cal = a.first_action_utc + np.floor(grid) * DAY             # TFA control stays a daily calendar measure
    f['tfa_posts_30d'] = np.searchsorted(tfa, cal) - np.searchsorted(tfa, cal - 30 * DAY)
    rows.append(pd.DataFrame(f))
ex = pd.concat(rows, ignore_index=True)
print(f'episodes {len(ex)}  accounts {ex.id.nunique()}  {time.time() - t0:.0f}s', flush=True)
assert (ex.groupby('id').event.sum() == 1).all()
save_new(ex, ('test_' if LIMIT else '') + (f'exact_{VERSION}_to_cutoff.parquet' if TO_CUTOFF else f'exact_{VERSION}.parquet'))
