# WHAT: draws 100 r/TheseFuckingAccounts submissions for checking the username extraction by hand (v3 of the review sample).
#       Same extraction as build_accused_table.py; posts that use each new title/url rule are over-sampled by a quota.
# READS: the two submission corpora, cells 8-12 of collect_accused_usernames_and_first_mention.ipynb, exclude_helpers_and_victims.py, username_shape.py.
# WRITES: data/extraction_sample_100_v3.csv (one row per sampled post), and collection_scripts/sample3.json.
# RUN: python extraction_review_sample_100.py
import os, sys, re, json, csv, random, collections, pandas as pd
# exclude_helpers_and_victims.py and username_shape.py sit next to this script in collection_scripts/.
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from exclude_helpers_and_victims import non_accused
from username_shape import looks_like_username
os.chdir('/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning')
RARE_MAX=3
# Load and stack both corpora, one copy per id, deduplicated on title + text + url (oldest kept).
COLS=['id','title','selftext','url','created_utc']
old=pd.read_csv('/data/Emily_data/TheseFuckingAccounts_submissions.csv',low_memory=False)[COLS+['url_overridden_by_dest']]
new=pd.read_parquet('data/tfa_submissions_2023_2026.parquet')[COLS+['url_overridden_by_dest']]
old['source']='csv_2014_2022'; new['source']='arctic_2023_2026'
both=pd.concat([old,new],ignore_index=True)
for k in ('title','selftext','url','url_overridden_by_dest'): both[k]=both[k].fillna('')
both['created_utc']=both.created_utc.astype('int64')
both=(both.drop_duplicates('id').sort_values('created_utc')
        .drop_duplicates(['title','selftext','url'],keep='first').reset_index(drop=True))
# Base extraction rules: cells 8-12 of the notebook, executed into namespace g.
nb=json.load(open('collect_accused_usernames_and_first_mention.ipynb'))
g={'re':re,'pd':pd}
for i in (8,9,10,11,12):
    s=''.join(nb['cells'][i]['source'])
    if i==11: s=s.split('VOCABULARY =')[0]
    exec(compile(s,f'c{i}','exec'),g)
ex=g['extract_suspects']; unesc=g['unescape_markdown']; RES=g['RESERVED_NAMES']
LEAD=g['TITLE_LEAD_RE']; ok_bare=g['_acceptable_bare_name']; fv=g['find_victims']
VOC=g['build_vocabulary'](both)
PROFILE=re.compile(r'(?:https?://)?(?:[a-z0-9-]+\.)*reddit\.com/u(?:ser)?/([A-Za-z0-9_-]{3,20})(?![A-Za-z0-9_-])',re.I)
COMMA_LEAD=re.compile(r'^\s*([A-Za-z0-9_-]{3,20})\s*,\s+')
WOKE=re.compile(r'^\s*([A-Za-z0-9_-]{3,20})\s*[,:—–-]*\s*(?:just\s+)?(?:woke|born|was\s+born)',re.I)
LEAD_RUN=re.compile(r'^\s*((?:[A-Za-z0-9_-]{3,20}\s*,\s*)+(?:and\s+)?[A-Za-z0-9_-]{3,20})(?![A-Za-z0-9_-])')
POSS=re.compile(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_-]{3,20})['\u2019]s\b")
POSS_NOUN=re.compile(r'^\s*(?:\d+\s+\w+\s+old\s+)?(?:account|profile|alt|bot|post|posts|comment|comments|submission|submissions|history|karma|page|spam)',re.I)
# Prose frequency of each word (u/ mentions and URLs removed): rare words can be usernames.
PROSE=re.compile(r'(?<![A-Za-z0-9_./])/?u/[A-Za-z0-9_-]{3,20}|https?://\S+|www\.\S+')
freq=collections.Counter()
for t,b in zip(both.title,both.selftext):
    freq.update(w.lower() for w in re.findall(r'[A-Za-z]{3,20}',PROSE.sub(' ',(t or '')+' '+(b or ''))))
# Title made only of username-shaped, rare tokens.
def ulist(title):
    toks=[t.strip('.,;:()[]\'"') for t in title.split()]; toks=[t for t in toks if t]
    if not toks or len(toks)>12: return []
    if not all(looks_like_username(t,freq,RARE_MAX) for t in toks): return []
    return toks
# Extraction for one post, same rules and order as build_accused_table.py.
# Returns (kept names, names dropped as helper/victim, victims named in the post).
def analyse(p):
    roles=non_accused(unesc(p.title))|non_accused(unesc(p.selftext))
    found=ex(p.title,p.selftext,VOC)
    mrun=LEAD_RUN.search(p.title)
    if mrun:
        tk=[t.strip() for t in re.split(r',|\band\b',mrun.group(1)) if t.strip()]
        if len(tk)>=2 and all(looks_like_username(t,freq,RARE_MAX) for t in tk):
            for nm in tk:
                k=nm.lower()
                if k not in RES and k not in found: found[k]=('title_lead_run',nm)
    for mp in POSS.finditer(p.title):
        nm=mp.group(1); k=nm.lower()
        if p.title[max(0,mp.start()-2):mp.start()].lower().endswith('u/'): continue
        if not looks_like_username(nm,freq,RARE_MAX): continue
        if not POSS_NOUN.match(p.title[mp.end():]): continue
        if k not in RES and k not in found: found[k]=('title_possessive',nm)
    for nm in ulist(p.title):
        k=nm.lower()
        if k not in RES and k not in found: found[k]=('title_username_list',nm)
    mc=COMMA_LEAD.search(p.title)
    if mc:
        nm=mc.group(1); k=nm.lower()
        if ok_bare(nm) and k not in RES and k not in found and looks_like_username(nm,freq,RARE_MAX):
            found[k]=('title_lead_comma',nm)
    mw=WOKE.search(p.title)
    if mw:
        nm=mw.group(1); k=nm.lower()
        if ok_bare(nm) and k not in RES and k not in found and freq[k]<RARE_MAX: found[k]=('title_woke',nm)
    ml=LEAD.search(p.title)
    if ml:
        nm=ml.group(1); k=nm.lower()
        if ok_bare(nm) and k not in RES and k not in found and freq[k]<RARE_MAX: found[k]=('title_lead_rare',nm)
    for field in (p.url,p.url_overridden_by_dest):
        for nm in PROFILE.findall(field or ''):
            k=nm.lower()
            if k in RES or k in roles: continue
            if k not in found: found[k]=('submission_url',nm)
    kept={k:v for k,v in found.items() if k not in roles}
    return kept, sorted(set(found)&roles), sorted(fv(unesc(p.title))|fv(unesc(p.selftext)))

recs=[]
for p in both.itertuples(index=False):
    kept,dropped,vics=analyse(p)
    recs.append((p,kept,dropped,vics))
# Rules added on top of the notebook's base rules.
NEW={'submission_url','title_lead_rare','title_lead_comma','title_username_list','title_woke','title_lead_run','title_possessive'}
# Post indexes grouped by the first new rule they use.
bynew=collections.defaultdict(list)
for i,(p,k,d,v) in enumerate(recs):
    for m,_ in k.values():
        if m in NEW: bynew[m].append(i); break
# Fixed seed so the same 100 posts come back. Quota per new rule, then random posts up to 100, in date order.
random.seed(2026)
picked=[]
quota={'submission_url':12,'title_lead_rare':12,'title_lead_comma':7,'title_username_list':7,
       'title_lead_run':6,'title_possessive':5,'title_woke':2}
for m,n in quota.items():
    pool=bynew.get(m,[]); random.shuffle(pool); picked+=pool[:min(n,len(pool))]
picked=list(dict.fromkeys(picked))
rest=[i for i in range(len(recs)) if i not in set(picked)]
random.shuffle(rest); picked+=rest[:100-len(picked)]
picked.sort(key=lambda i: recs[i][0].created_utc)
# Post text longer than 3,000 characters is cut in the review file.
CAP=3000
rows=[]
for i in picked:
    p,kept,dropped,vics=recs[i]
    ms=[m for m,_ in kept.values()]
    rows.append({'post_id':p.id,'post_date':pd.to_datetime(p.created_utc,unit='s').strftime('%Y-%m-%d %H:%M'),
      'source':p.source,'title':p.title,
      'post_text':(p.selftext[:CAP]+'  …[TRUNCATED]') if len(p.selftext)>CAP else p.selftext,
      'post_text_chars':len(p.selftext),'truncated':len(p.selftext)>CAP,
      'submission_url':p.url,
      'n_suspicious':len(kept),'suspicious_names':' | '.join(w for _,w in kept.values()),
      'extraction_method':' | '.join(ms),
      'uses_new_rule':any(m in NEW for m in ms),
      'n_excluded_victim':len(vics),'excluded_as_victim':' | '.join(vics),
      'n_excluded_role':len(dropped),'excluded_as_helper_or_victim':' | '.join(dropped)})
out=pd.DataFrame(rows)
out.to_csv('data/extraction_sample_100_v3.csv',index=False,quoting=csv.QUOTE_ALL)
print('wrote data/extraction_sample_100_v3.csv')
print(f'{len(out)} posts | {out.n_suspicious.sum()} names extracted | {out.uses_new_rule.sum()} posts use a new rule')
print(collections.Counter(m for r in rows for m in r['extraction_method'].split(' | ') if m))
print('zero-name posts:',int((out.n_suspicious==0).sum()),'| victims excluded:',int((out.n_excluded_victim>0).sum()),
      '| roles excluded:',int((out.n_excluded_role>0).sum()))
# Same rows as JSON for a review page.
jr=[{'id':r['post_id'],'date':r['post_date'],'src':r['source'],'title':r['title'],'text':r['post_text'],
     'chars':r['post_text_chars'],'trunc':bool(r['truncated']),'url':r['submission_url'],
     'names':[x.strip() for x in r['suspicious_names'].split('|') if x.strip()],
     'methods':[x.strip() for x in r['extraction_method'].split('|') if x.strip()],
     'newrule':bool(r['uses_new_rule']),
     'victims':[x.strip() for x in r['excluded_as_victim'].split('|') if x.strip()],
     'roles':[x.strip() for x in r['excluded_as_helper_or_victim'].split('|') if x.strip()]} for r in rows]
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'sample3.json'),'w').write(json.dumps(jr,ensure_ascii=False))
