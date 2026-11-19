"""Test-run build: daily time-varying covariates for the 998 pilot accounts (subset of the agreed variables).

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../aft_model/); kept for the record.

One row per account-day d = 0 .. ceil(T)-1, where T = days from first archived action to first accusation. Each count
covers the previous 7 days (days d-7 .. d-1), so covariates are lagged by one day and never use the day itself.
Inputs: sample_998_v4_excl_output/ (accounts, posts, comments), subreddit_groups_v9.parquet, crosspost flags,
replies on own posts (replies_output/posts_parts/), and the TFA submissions CSV (calendar control).
Output: daily.parquet next to this script (id, start, stop, covariate columns).
Run: python build.py. build_exact.py execs the data-loading part of this file (everything above the group list), so
keep that part free of side effects beyond loading."""
import os, re, glob, time
import numpy as np
import pandas as pd

P2 = '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_analysis'
S = os.path.join(P2, 'sample_998_v4_excl_output')
DAY = 86400.0
WIN = 7                                             # look-back window in days

# host patterns: image/video hosts count as visual media; anything neither visual nor Reddit counts as an external link
VISUAL = re.compile(r'(^|\.)(i\.redd\.it|v\.redd\.it|i\.reddituploads\.com|imgur\.com|gfycat\.com|redgifs\.com|giphy\.com|'
                    r'streamable\.com|youtube\.com|youtu\.be)$')
REDDIT = re.compile(r'(^|\.)(reddit\.com|redd\.it|reddituploads\.com|redditmedia\.com)$|^self\.')
URL = re.compile(r'https?://([^/\s)\]>"\']+)', re.I)


def hosts(text):
    """Host names of all http(s) URLs in a text.

    text: str or None. Returns a list of lower-case hosts with any port and leading 'www.' removed."""
    return [h.lower().split(':')[0].removeprefix('www.') for h in URL.findall(text or '')]


def link_flags(host_lists):
    """Flag items that link to visual media or to external (non-Reddit, non-visual) sites.

    host_lists: one list of hosts per item. Returns two boolean arrays (visual, external) of the same length."""
    vis = np.array([any(VISUAL.search(h) for h in hs) for hs in host_lists])
    ext = np.array([any(not VISUAL.search(h) and not REDDIT.search(h) for h in hs) for hs in host_lists])
    return vis, ext


t0 = time.time()
acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet'))[
    ['username_key', 'username', 'first_action_utc', 'first_mention_adj_utc', 'days_to_accused']].set_index('username_key')
groups = pd.read_parquet(os.path.join(P2, 'subreddit_review_output', 'subreddit_groups_v9.parquet')).set_index('subreddit')['group']

posts = pd.read_parquet(os.path.join(S, 'sample_posts.parquet'), columns=[
    'id', 'outer_key', 'created_utc', 'subreddit', 'title', 'selftext', 'url', 'domain', 'is_self', 'over_18', 'score',
    'edited']).drop_duplicates(['outer_key', 'id'])
coms = pd.read_parquet(os.path.join(S, 'sample_comments.parquet'), columns=[
    'id', 'outer_key', 'created_utc', 'subreddit', 'link_id', 'body', 'score', 'edited']).drop_duplicates(['outer_key', 'id'])
for d in (posts, coms):
    d['acc_utc'] = d['outer_key'].map(acc['first_mention_adj_utc'])     # accusation time (end of follow-up)
    d['t0'] = d['outer_key'].map(acc['first_action_utc'])               # time origin
# keep only behaviour before the accusation
posts = posts[posts.created_utc < posts.acc_utc].copy()
coms = coms[coms.created_utc < coms.acc_utc].copy()

# links: post url host (non-self) + hosts in selftext/body
post_hosts = [([str(dm).lower().removeprefix('www.')] if not s else []) + hosts(st)
              for dm, s, st in zip(posts.domain, posts.is_self.astype(str).eq('True'), posts.selftext)]
posts['visual'], posts['external'] = link_flags(post_hosts)
posts.loc[posts.url.fillna('').str.contains('reddit.com/gallery/'), 'visual'] = True     # Reddit image galleries
coms['visual'], coms['external'] = link_flags([hosts(b) for b in coms.body])

xp = pd.read_parquet(os.path.join(P2, 'crosspost_output', 'crosspost_flags_998.parquet'), columns=['id', 'outer_key', 'native_crosspost', 'xpost_text'])
# manual x-post flag recomputed with URLs stripped (2 URL-parameter false positives)
XPOST = re.compile(r'(?<![a-z])(x-?\s?post(?:ed)?|cross-?\s?post(?:ed)?)(?![a-z])', re.I)
posts['xtext'] = (posts.title.fillna('') + ' ' + posts.selftext.fillna('').str[:500]).str.replace(r'https?://\S+', ' ', regex=True)
posts = posts.merge(xp[['id', 'outer_key', 'native_crosspost']], on=['id', 'outer_key'], how='left')
# crosspost = Reddit's native crosspost or an "x-post"/"cross-post" mention in the title or first 500 characters
posts['crosspost'] = posts.native_crosspost.fillna(False).astype(bool) | posts.xtext.str.contains(XPOST)

posts['group'] = posts.subreddit.map(groups)
coms['group'] = coms.subreddit.map(groups)
assert posts.group.notna().all() and coms.group.notna().all(), 'subreddit missing from v9 groups'
# NSFW post outside subreddits where NSFW is expected (adult groups and user profiles)
posts['nsfw_outside'] = posts.over_18.astype(str).eq('True') & ~posts.group.isin(['adult', 'profile'])
posts['score'] = pd.to_numeric(posts.score); coms['score'] = pd.to_numeric(coms.score)
# downvoted: posts at score <= 0, comments at score < 0; popular: score >= 100
posts['down'] = posts.score.le(0); coms['down'] = coms.score.lt(0)
posts['popular'] = posts.score.ge(100); coms['popular'] = coms.score.ge(100)
# thread id: a post is its own thread; a comment belongs to the post in link_id
posts['thread'] = posts['id']
coms['thread'] = coms.link_id.str.removeprefix('t3_')
for d in (posts, coms):
    d['day'] = np.floor((d.created_utc - d.t0) / DAY).astype(int)     # account-day of the item (0 = first day)
    # 'edited' is 'False' or an edit timestamp; an edit counts on the day it happened, if before the accusation
    e = pd.to_numeric(d.edited.replace('False', np.nan), errors='coerce')
    d['edit_day'] = np.where(e.notna() & (e < d['acc_utc']), np.floor((e - d.t0) / DAY), np.nan)

# dated replies received on own posts (post threads done; comment replies still collecting -> not in the test run)
rep = pd.concat(pd.read_parquet(f, columns=['author', 'created_utc', 'source_id', 'outer_key'])
                for f in glob.glob(os.path.join(P2, 'replies_output', 'posts_parts', 'replies_*.parquet')))
rep = rep[rep.author.str.lower() != rep.outer_key]                     # drop the account's own replies
rep['acc_utc'] = rep.outer_key.map(acc['first_mention_adj_utc']); rep['t0'] = rep.outer_key.map(acc['first_action_utc'])
rep = rep[pd.to_numeric(rep.created_utc) < rep['acc_utc']]
rep['day'] = np.floor((pd.to_numeric(rep.created_utc) - rep.t0) / DAY).astype(int)

# TFA posts in the previous 30 days (calendar), 2014-2022 corpus
tfa = np.sort(pd.read_csv('/data/Emily_data/TheseFuckingAccounts_submissions.csv', usecols=['created_utc'])
              .created_utc.pipe(pd.to_numeric, errors='coerce').dropna().to_numpy())
print(f'loaded {time.time() - t0:.0f}s', flush=True)

GROUPS = ['profile', 'adult', 'crypto', 'politics', 'karma_farming', 'general', 'news', 'sales', 'location', 'entertainment', 'other']


def window_counts(days, n):
    """items per day -> count in the previous WIN days (days < d) for d = 0..n-1

    days: int array of item days; n: number of account-days. Returns an int array of length n."""
    c = np.bincount(days[(days >= 0) & (days < n)], minlength=n) if len(days) else np.zeros(n, int)
    C = np.r_[0, np.cumsum(c)]                      # C[d] = items on days < d
    d = np.arange(n)
    return C[d] - C[np.maximum(d - WIN, 0)]         # items on days d-WIN .. d-1


def thread_windows(thread_days, n):
    """thread_days: DataFrame(thread, day) -> number of distinct threads with activity in [d-WIN, d-1]

    n: number of account-days. Returns an int array of length n."""
    # difference array: +1 on the first day a thread is in the window, -1 on the first day it has left
    diff = np.zeros(n + WIN + 2, int)
    for _, g in thread_days.groupby('thread'):
        dd = np.unique(g.day.to_numpy())
        # split the thread's active days into blocks; a gap > WIN days means the thread drops out of the window
        block = np.r_[0, np.cumsum(np.diff(dd) > WIN)]
        for b in np.unique(block):
            s, e = dd[block == b][[0, -1]]
            diff[s + 1] += 1; diff[e + WIN + 1] -= 1   # block with days s..e is counted on days s+1 .. e+WIN
    return np.cumsum(diff)[:n]


rows = []
pg, cg, rg = posts.groupby('outer_key'), coms.groupby('outer_key'), rep.groupby('outer_key')
empty_p, empty_c = posts.iloc[:0], coms.iloc[:0]
for k, a in acc.iterrows():
    T = a.days_to_accused
    n = int(np.ceil(T))
    p = pg.get_group(k) if k in pg.groups else empty_p
    c = cg.get_group(k) if k in cg.groups else empty_c
    r = rg.get_group(k) if k in rg.groups else rep.iloc[:0]
    act = pd.concat([p[['thread', 'day', 'group', 'subreddit']], c[['thread', 'day', 'group', 'subreddit']]])
    # one row per day; the last day stops at T (a fraction of a day)
    f = {'id': np.repeat(k, n), 'start': np.arange(n, dtype=float), 'stop': np.minimum(np.arange(1, n + 1), T)}
    for gname in GROUPS:
        f[f'thr_{gname}'] = thread_windows(act[act.group == gname], n)
    # flooding: share of last-7-day threads in the single most-used subreddit
    tot = np.zeros(n); mx = np.zeros(n)
    for _, s in act.groupby('subreddit'):
        w = thread_windows(s, n); tot += w; mx = np.maximum(mx, w)
    f['flooding'] = np.where(tot > 0, mx / np.maximum(tot, 1), 0.0)
    f['links_external'] = window_counts(p.day[p.external].to_numpy(), n) + window_counts(c.day[c.external].to_numpy(), n)
    f['visual_media'] = window_counts(p.day[p.visual].to_numpy(), n) + window_counts(c.day[c.visual].to_numpy(), n)
    f['crossposts'] = window_counts(p.day[p.crosspost].to_numpy(), n)
    f['nsfw_outside'] = window_counts(p.day[p.nsfw_outside].to_numpy(), n)
    f['edits'] = (window_counts(p.edit_day.dropna().astype(int).to_numpy(), n)
                  + window_counts(c.edit_day.dropna().astype(int).to_numpy(), n))
    f['down_posts'] = window_counts(p.day[p.down].to_numpy(), n)
    f['down_comments'] = window_counts(c.day[c.down].to_numpy(), n)
    f['popular_posts'] = window_counts(p.day[p.popular].to_numpy(), n)
    f['popular_comments'] = window_counts(c.day[c.popular].to_numpy(), n)
    f['replies_on_posts'] = window_counts(r.day.to_numpy(), n)
    first_p = p.day.min() if len(p) else np.inf
    first_c = c.day.min() if len(c) else np.inf
    # 1 from the day after the first post / comment (1-day lag, same as the counts)
    f['has_posted'] = (np.arange(n) > first_p).astype(float)
    f['has_commented'] = (np.arange(n) > first_c).astype(float)
    # TFA posts in the 30 calendar days before the start of each account-day
    cal = a.first_action_utc + np.arange(n) * DAY
    f['tfa_posts_30d'] = np.searchsorted(tfa, cal) - np.searchsorted(tfa, cal - 30 * DAY)
    rows.append(pd.DataFrame(f))
daily = pd.concat(rows, ignore_index=True)
print(f'daily rows {len(daily)}  {time.time() - t0:.0f}s', flush=True)
daily.to_parquet(os.path.join(os.path.dirname(__file__), 'daily.parquet'))
