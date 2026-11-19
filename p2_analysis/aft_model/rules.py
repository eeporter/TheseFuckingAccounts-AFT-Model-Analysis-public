"""Variable and sample rules the user decided for the AFT model. The build imports these so every rule lives in one place.

Decided 2026-10-15 (user):
- Flooding needs 10+ threads in the busiest subreddit in the last 7 days; below that it is 0.
- Visual media = an image or video that would have shown up in the thread or as part of the post. Those links are
  not also counted as URL links. A media link that would only have shown as a link counts as a URL link.
- Accounts whose capped archive (10,000 posts or comments) ends before the accusation are dropped.
Decided 2026-10-16 (user):
- Every capped account is dropped (11 in the pilot), not only those whose archive ends before the accusation.
- Right truncation: fixed 1-year window (HORIZON_DAYS): first action at least a year before the cutoff, accused within a year.

Library module, reads and writes no files; imported by build.py (flooding, item_link_flags, capped_accounts, CUTOFF)
and fit_main.py / fit_full.py (in_window, HORIZON_DAYS).
"""
import re
import numpy as np
import pandas as pd

FLOOD_MIN_THREADS = 10
CUTOFF = 1669766400          # pool rule 5: accused before 2022-11-30 00:00 UTC (kept: before mainstream AI)
HORIZON_DAYS = 365


def flooding(busiest, total):
    """busiest subreddit's share of the window's threads, only when it has FLOOD_MIN_THREADS+ threads, else 0

    busiest: thread count in the account's busiest subreddit in the 7-day window; total: all its threads in the window
    (array-likes, same length). Returns a float array in [0, 1].
    """
    busiest, total = np.asarray(busiest, float), np.asarray(total, float)
    return np.where(busiest >= FLOOD_MIN_THREADS, busiest / np.maximum(total, 1), 0.0)   # max(total, 1): no 0/0


# --- visual media: does the image or video render? ---------------------------------------------------------------
# Link posts (where='link') show their media in the post: Reddit-hosted media and galleries, the embed hosts
# (link_type 'visual' in link_domain_types_v1), other image/video hosts (link_type 'media_host') and direct media files on
# any host. File lockers labelled media_host only give a download page, so they stay URL links.
# Links in selftext or comment text showed as links on Reddit before 2022-11-30, except images uploaded into the text
# (preview.redd.it) and image/GIF markdown (![img](...), ![gif](giphy|...)).
LOCKERS = {'mega.nz', 'mixdrop.co', 'verystream.com'}
# MEDIA_FILE: a media file extension at the end of the URL path (before any query or fragment)
MEDIA_FILE = re.compile(r'\.(?:jpe?g|png|gif|gifv|webp|bmp|mp4|webm|mov)(?:[?#]|$)', re.I)
INLINE_HOSTS = {'preview.redd.it'}
INLINE_MARKDOWN = re.compile(r'!\[(?:img|gif)\]\(', re.I)


def link_renders(links):
    """links: rows of link_domain_output/links_pre_accusation_v1.parquet -> bool Series, True if the media showed

    Needs columns kind, where, link_type, domain, host, url (as built by links.build). The same rule is applied to any
    link table with those columns.
    """
    link_post = links['kind'].eq('post') & links['where'].eq('link')     # the post's own link, not a link in its text
    media = (links['link_type'].eq('visual') | (links['link_type'].eq('media_host') & ~links['domain'].isin(LOCKERS))
             | links['url'].str.contains(MEDIA_FILE))
    # link posts: rendered if the target is media; links inside text (selftext, comments): only uploaded images
    return (link_post & media) | (~link_post & links['host'].isin(INLINE_HOSTS))


def item_link_flags(links, items, kind, text_col):
    """items: posts or comments (outer_key, id, text_col) -> (visual, external) bool arrays aligned with items.

    visual: any link (or image markdown) that showed the media. external: any other link off Reddit.
    links: link table (links.build); kind: 'post' or 'comment'; text_col: 'selftext' for posts, 'body' for comments.
    An item can be both visual and external if it has one link of each kind; a rendered link is never external.
    """
    L = links[links['kind'].eq(kind)].copy()
    L['renders'] = link_renders(L)
    L['offsite'] = ~L['renders'] & ~L['link_type'].isin(['reddit', 'reddit/visual'])   # links to Reddit are not URL links
    g = L.groupby(['outer_key', 'item_id'])[['renders', 'offsite']].any()     # one row per item: any such link
    key = pd.MultiIndex.from_arrays([items['outer_key'], items['id']])
    vis = g['renders'].reindex(key, fill_value=False).to_numpy()            # items with no links -> False
    ext = g['offsite'].reindex(key, fill_value=False).to_numpy()
    vis |= items[text_col].fillna('').str.contains(INLINE_MARKDOWN).to_numpy()   # image/GIF markdown has no URL to match
    return vis, ext


# --- sample: drop every account whose archive hit the cap ----------------------------------------------------------
CAP = 10_000   # per-stream (posts or comments) limit of an account's archive pull


def capped_before_accusation(acc, posts_all, coms_all):
    """accounts with a stream at the cap whose last archived item is before the (edit-adjusted) accusation, i.e. whose
    pre-accusation history is incomplete (7 in the pilot; the first rule, 2026-10-15)

    Superseded by capped_accounts (2026-10-16). acc: accounts indexed by outer_key with first_mention_adj_utc (epoch s);
    posts_all, coms_all: all archived items (outer_key, id, created_utc). Returns a sorted list of outer_keys.
    """
    out = set()
    for d in (posts_all, coms_all):
        g = d.drop_duplicates(['outer_key', 'id']).groupby('outer_key').created_utc.agg(['size', 'max'])
        g = g[g['size'] >= CAP]
        out |= set(g.index[g['max'] < acc['first_mention_adj_utc'].reindex(g.index)])
    return sorted(out)


def capped_accounts(acc, posts_all, coms_all):
    """every account flagged history_truncated with a stream at the cap, whether or not the archive reaches the
    accusation (user, 2026-10-16: drop all capped accounts; 11 in the pilot). Accounts with more than 10,000 items whose
    pull was complete (history_truncated False) are kept.

    acc: accounts indexed by outer_key with history_truncated; posts_all, coms_all: all archived items (outer_key, id).
    Returns a sorted list of outer_keys to drop.
    """
    trunc = set(acc.index[acc['history_truncated'].fillna(False).astype(bool)])
    out = set()
    for d in (posts_all, coms_all):
        n = d.drop_duplicates(['outer_key', 'id']).groupby('outer_key').size()   # items per account in this stream
        out |= set(n.index[n >= CAP]) & trunc
    return sorted(out)


def in_window(acc, H=HORIZON_DAYS):
    """accounts in the fixed-horizon sample: first action >= H days before the cutoff and accused within H days

    acc: accounts with first_action_utc (epoch seconds) and days_to_accused (days); H in days. This is the 2026-10-16
    right-truncation rule: every kept account had at least H days between its first action and the cutoff.
    Returns the index of the accounts kept.
    """
    tau = (CUTOFF - acc['first_action_utc']) / 86400.0      # days from first action to the cutoff
    return acc.index[(tau >= H) & (acc['days_to_accused'] <= H)]
