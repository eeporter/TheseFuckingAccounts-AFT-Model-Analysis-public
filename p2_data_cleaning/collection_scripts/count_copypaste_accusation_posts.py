# WHAT: one-off check of how often accusation posts say the account copied someone else's content,
#       and how many accused accounts would carry a 'copy-paste' flag if that were used.
# READS: /data/Emily_data/TheseFuckingAccounts_submissions.csv and cells 8-12 of collect_accused_usernames_and_first_mention.ipynb.
# WRITES: collection_scripts/posts.pkl (run from p2_data_cleaning/); prints counts.
# RUN: python count_copypaste_accusation_posts.py from p2_data_cleaning/.
import json, re, pandas as pd
# Load the notebook's extraction rules by executing cells 8-12 into the namespace g.
nb=json.load(open('collect_accused_usernames_and_first_mention.ipynb'))
g={"re":re,"pd":pd}
for i in (8,9,10,11,12):
    src=''.join(nb['cells'][i]['source'])
    if i==11: src=src.split('VOCABULARY =')[0]   # defer vocab build
    exec(compile(src,f'cell{i}','exec'),g)
# 2014-2022 submissions, deduplicated on title + text, oldest kept.
posts=pd.read_csv('/data/Emily_data/TheseFuckingAccounts_submissions.csv',low_memory=False)[['id','title','selftext','created_utc']]
posts['title']=posts['title'].fillna(''); posts['selftext']=posts['selftext'].fillna('')
posts=posts.sort_values('created_utc').drop_duplicates(subset=['title','selftext'],keep='first').reset_index(drop=True)
VOC=g['build_vocabulary'](posts)
ex=g['extract_suspects']
# Number of accused names the base rules extract from each post.
posts['n']=[len(ex(t,s,VOC)) for t,s in zip(posts.title,posts.selftext)]
pd.to_pickle((posts,),'collection_scripts/posts.pkl')

# Helper for reading example posts by hand.
def show(row):
    print('='*70); print('post id:',row.id)
    print('TITLE   :',row.title[:300])
    body=row.selftext[:700]
    print('SELFTEXT:',body if body.strip() else '(empty)')
import re
txt=(posts.title+' \n '+posts.selftext)
# Three signals that a post is about copied content; the None entries are computed separately below.
pat={
 'copy language (copied/copy-paste/stolen/repost)': r'\b(copied|copy[\s/]?pas|stolen|steal|repost|plagiar)',
 'victim named (possessive or "from u/x")': None,
 'links to a source reddit permalink': r'reddit\.com/r/[A-Za-z0-9_]+/comments/[a-z0-9]{5,8}',
 'both copy language AND source permalink': None,
}
has_copy=txt.str.contains(pat['copy language (copied/copy-paste/stolen/repost)'],case=False,regex=True)
has_link=txt.str.contains(pat['links to a source reddit permalink'],case=False,regex=True)
# Victim named: the notebook's possessive/attribution rule finds a copied-from account in the title or text.
vic=[bool(g['find_victims'](g['unescape_markdown'](t))|g['find_victims'](g['unescape_markdown'](s))) for t,s in zip(posts.title,posts.selftext)]
vic=pd.Series(vic)
n=len(posts)
# Share of posts with each signal and combinations of them.
for lab,m in [('copy/repost language',has_copy),('victim account named',vic),('source permalink present',has_link),('copy language AND permalink',has_copy&has_link),('all three',has_copy&has_link&vic)]:
    print(f'{lab:34} {m.sum():>6,} posts  ({m.mean():5.1%})')
# how many *accounts* would inherit a copy flag
# Accounts named in any post with copy language or a named victim.
accs=set(); flagged=set()
for row,f in zip(posts.itertuples(index=False), (has_copy|vic).values):
    d=ex(row.title,row.selftext,VOC)
    accs.update(d)
    if f: flagged.update(d)
print(f'\naccounts total {len(accs):,} | would carry a copy-paste flag {len(flagged):,} ({len(flagged)/len(accs):.1%})')
