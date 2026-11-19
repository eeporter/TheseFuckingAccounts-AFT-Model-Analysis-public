"""Variable and sample rules the user decided for the AFT model. The build imports these so every rule lives in one place.

Decided 2026-10-15 (user):
- Flooding needs 10+ threads in the busiest subreddit in the last 7 days; below that it is 0.
- Visual media = an image or video that would have shown up in the thread or as part of the post. Those links are
  not also counted as URL links. A media link that would only have shown as a link counts as a URL link.
- Accounts whose capped archive (10,000 posts or comments) ends before the accusation are dropped.
Decided 2026-10-16 (user):
- Every capped account is dropped (11 in the pilot), not only those whose archive ends before the accusation.
- Right truncation: fixed 1-year window (HORIZON_DAYS): first action at least a year before the cutoff, accused within a year.
"""
import re
import numpy as np
import pandas as pd

FLOOD_MIN_THREADS = 10
CUTOFF = 1669766400          # pool rule 5: accused before 2022-11-30 00:00 UTC (kept: before mainstream AI)
HORIZON_DAYS = 365


def flooding(busiest, total):
    """busiest subreddit's share of the window's threads, only when it has FLOOD_MIN_THREADS+ threads, else 0"""
    busiest, total = np.asarray(busiest, float), np.asarray(total, float)
    return np.where(busiest >= FLOOD_MIN_THREADS, busiest / np.maximum(total, 1), 0.0)


# --- visual media: does the image or video render? ---------------------------------------------------------------
# Link posts (where='link') show their media in the post: Reddit-hosted media and galleries, the embed hosts
# (link_type 'visual' in Link_domains_v1), other image/video hosts (link_type 'media_host') and direct media files on
# any host. File lockers labelled media_host only give a download page, so they stay URL links.
# Links in selftext or comment text showed as links on Reddit before 2022-11-30, except images uploaded into the text
# (preview.redd.it) and image/GIF markdown (![img](...), ![gif](giphy|...)).
LOCKERS = {'mega.nz', 'mixdrop.co', 'verystream.com'}
MEDIA_FILE = re.compile(r'\.(?:jpe?g|png|gif|gifv|webp|bmp|mp4|webm|mov)(?:[?#]|$)', re.I)
INLINE_HOSTS = {'preview.redd.it'}
INLINE_MARKDOWN = re.compile(r'!\[(?:img|gif)\]\(', re.I)


def link_renders(links):
    """links: rows of link_domain_output/links_pre_accusation_v1.parquet -> bool Series, True if the media showed"""
    link_post = links['kind'].eq('post') & links['where'].eq('link')
    media = (links['link_type'].eq('visual') | (links['link_type'].eq('media_host') & ~links['domain'].isin(LOCKERS))
             | links['url'].str.contains(MEDIA_FILE))
    return (link_post & media) | (~link_post & links['host'].isin(INLINE_HOSTS))


def item_link_flags(links, items, kind, text_col):
    """items: posts or comments (outer_key, id, text_col) -> (visual, external) bool arrays aligned with items.

    visual: any link (or image markdown) that showed the media. external: any other link off Reddit."""
    L = links[links['kind'].eq(kind)].copy()
    L['renders'] = link_renders(L)
    L['offsite'] = ~L['renders'] & ~L['link_type'].isin(['reddit', 'reddit/visual'])
    g = L.groupby(['outer_key', 'item_id'])[['renders', 'offsite']].any()
    key = pd.MultiIndex.from_arrays([items['outer_key'], items['id']])
    vis = g['renders'].reindex(key, fill_value=False).to_numpy()
    ext = g['offsite'].reindex(key, fill_value=False).to_numpy()
    vis |= items[text_col].fillna('').str.contains(INLINE_MARKDOWN).to_numpy()
    return vis, ext


# --- sample: drop every account whose archive hit the cap ----------------------------------------------------------
CAP = 10_000


def capped_before_accusation(acc, posts_all, coms_all):
    """accounts with a stream at the cap whose last archived item is before the (edit-adjusted) accusation, i.e. whose
    pre-accusation history is incomplete (7 in the pilot; the first rule, 2026-10-15)"""
    out = set()
    for d in (posts_all, coms_all):
        g = d.drop_duplicates(['outer_key', 'id']).groupby('outer_key').created_utc.agg(['size', 'max'])
        g = g[g['size'] >= CAP]
        out |= set(g.index[g['max'] < acc['first_mention_adj_utc'].reindex(g.index)])
    return sorted(out)


def capped_accounts(acc, posts_all, coms_all):
    """every account flagged history_truncated with a stream at the cap, whether or not the archive reaches the
    accusation (user, 2026-10-16: drop all capped accounts; 11 in the pilot). Accounts with more than 10,000 items whose
    pull was complete (history_truncated False) are kept."""
    trunc = set(acc.index[acc['history_truncated'].fillna(False).astype(bool)])
    out = set()
    for d in (posts_all, coms_all):
        n = d.drop_duplicates(['outer_key', 'id']).groupby('outer_key').size()
        out |= set(n.index[n >= CAP]) & trunc
    return sorted(out)


def in_window(acc, H=HORIZON_DAYS):
    """accounts in the fixed-horizon sample: first action >= H days before the cutoff and accused within H days"""
    tau = (CUTOFF - acc['first_action_utc']) / 86400.0
    return acc.index[(tau >= H) & (acc['days_to_accused'] <= H)]
