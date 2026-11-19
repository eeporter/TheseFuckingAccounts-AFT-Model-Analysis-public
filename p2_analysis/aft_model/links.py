"""Link table for any set of posts/comments, built the same way as Link_domains_v1.ipynb (which covers pre-accusation
items only). Domain types come from the rules there and the agent labels in link_domain_output/; domains the labels
never saw get link_type 'unlabeled' (counted as URL links unless they render)."""
import os, re
import pandas as pd
import tldextract

P2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISUAL_HOSTS = ['i.redd.it', 'v.redd.it', 'preview.redd.it', 'i.reddituploads.com', 'i.redditmedia.com']
VISUAL_DOMAINS = ['imgur.com', 'gfycat.com', 'redgifs.com', 'giphy.com', 'streamable.com', 'youtube.com', 'youtu.be']
REDDIT_DOMAINS = ['reddit.com', 'redd.it', 'redditmedia.com', 'reddituploads.com']
MESSAGING_DOMAINS = ['discord.gg', 'discord.com', 'discordapp.com', 't.me', 'telegram.me', 'telegram.org', 'whatsapp.com', 'wa.me']
OVERRIDES = {'https': 'other'}                    # Link_domains_v1 OVERRIDES
URL = re.compile(r'https?://[^\s)\]>"\'<]+', re.I)
_ext = tldextract.TLDExtract(suffix_list_urls=())  # offline: bundled public-suffix list


def build(posts, comments):
    """posts: outer_key, id, created_utc, url, is_self, selftext; comments: outer_key, id, created_utc, body"""
    recs = []
    for r in posts.itertuples():
        urls = [('text', u) for u in URL.findall(r.selftext or '')]
        if str(r.is_self) != 'True' and r.url:
            urls.append(('link', r.url))
        recs += [(r.outer_key, r.id, 'post', where, r.created_utc, u) for where, u in urls]
    for r in comments.itertuples():
        recs += [(r.outer_key, r.id, 'comment', 'text', r.created_utc, u) for u in URL.findall(r.body or '')]
    L = pd.DataFrame(recs, columns=['outer_key', 'item_id', 'kind', 'where', 'created_utc', 'url'])
    L['host'] = (L['url'].str.extract(r'https?://([^/?#]+)', flags=re.I)[0].str.lower().str.split(':').str[0]
                 .str.removeprefix('www.').fillna(''))
    L['domain'] = [(e.domain + '.' + e.suffix) if e.suffix else e.domain for e in map(_ext, L['host'])]
    labels = pd.read_csv(os.path.join(P2, 'link_domain_output', 'link_domain_llm_labels_v1.csv')).set_index('domain')['code']
    L['link_type'] = L['domain'].map(labels).fillna('unlabeled')
    for dom, t in OVERRIDES.items():
        L.loc[L['domain'].eq(dom), 'link_type'] = t
    rule = {'messaging': L['domain'].isin(MESSAGING_DOMAINS),
            'reddit': L['domain'].isin(REDDIT_DOMAINS) | (L['domain'].eq('') & L['url'].str.startswith('/')),
            'other': L['domain'].eq('') & ~L['url'].str.startswith('/'),
            'visual': (L['host'].isin(VISUAL_HOSTS) | L['domain'].isin(VISUAL_DOMAINS)
                       | (L['domain'].eq('reddit.com') & L['url'].str.contains('/gallery/', regex=False)))}
    for t in ['other', 'messaging', 'reddit', 'visual']:      # later wins: a gallery is visual, not reddit
        L.loc[rule[t], 'link_type'] = t
    return L
