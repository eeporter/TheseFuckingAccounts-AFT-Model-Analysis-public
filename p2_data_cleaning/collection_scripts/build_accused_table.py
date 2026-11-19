# WHAT: builds the final table of accused accounts (one row per account) from every r/TheseFuckingAccounts submission.
# READS: /data/Emily_data/TheseFuckingAccounts_submissions.csv (2014-2022), data/tfa_submissions_2023_2026.parquet (Arctic Shift),
#        cells 8-12 of collect_accused_usernames_and_first_mention.ipynb (the base extraction rules), exclude_helpers_and_victims.py, username_shape.py.
# WRITES: data/accused_accounts_final.parquet (overwritten) and data/excluded_role_mentions.parquet.
# RUN: python build_accused_table.py (from any folder; it changes into p2_data_cleaning/). Methods: DATA_COLLECTION.md section 1.
"""Final accused-account table.
Sources: submission title, selftext, AND the submission url (a link post whose url is a
Reddit profile IS the accusation - the title often never names the account).
Role exclusion (helper / copied-from victim) applied per post."""
import os, sys, re, json, pandas as pd
# exclude_helpers_and_victims.py and username_shape.py sit next to this script in collection_scripts/.
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from exclude_helpers_and_victims import non_accused
from username_shape import looks_like_username
os.chdir('/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning')

# --- 1. Load both submission corpora and stack them ---
COLS=['id','title','selftext','url','created_utc']
old=pd.read_csv('/data/Emily_data/TheseFuckingAccounts_submissions.csv',low_memory=False)[COLS+['url_overridden_by_dest']]
new=pd.read_parquet('data/tfa_submissions_2023_2026.parquet')[COLS+['url_overridden_by_dest']]
old['source']='csv_2014_2022'; new['source']='arctic_2023_2026'
both=pd.concat([old,new],ignore_index=True)
# Missing text fields become '' so the regexes below never see NaN.
for k in ('title','selftext','url','url_overridden_by_dest'): both[k]=both[k].fillna('')
both['created_utc']=both.created_utc.astype('int64')
# The same submission can be in both corpora; keep one copy per id.
both=both.drop_duplicates('id')
# dedupe key now includes url: two link posts can share a title but name different accounts
n0=len(both)
old_key=len(both.drop_duplicates(['title','selftext']))
both=both.sort_values('created_utc').drop_duplicates(['title','selftext','url'],keep='first').reset_index(drop=True)
print(f'corpus {n0:,} -> {len(both):,} (dedupe on title+selftext+url; the old title+selftext key would have left {old_key:,}, '
      f'collapsing {len(both)-old_key:,} distinct link posts)',flush=True)

# --- 2. Load the notebook's base extraction rules by executing its cells 8-12 into the namespace g ---
# Cell 11 is cut before VOCABULARY = so the vocabulary is built here from the combined corpus instead.
nb=json.load(open('collect_accused_usernames_and_first_mention.ipynb'))
g={'re':re,'pd':pd}
for i in (8,9,10,11,12):
    s=''.join(nb['cells'][i]['source'])
    if i==11: s=s.split('VOCABULARY =')[0]
    exec(compile(s,f'c{i}','exec'),g)
ex=g['extract_suspects']; unesc=g['unescape_markdown']; RES=g['RESERVED_NAMES']
RARE_MAX=3   # a lead name appearing <3 times as ordinary prose in the corpus is distinctive
PROFILE=re.compile(r'(?:https?://)?(?:[a-z0-9-]+\.)*reddit\.com/u(?:ser)?/([A-Za-z0-9_-]{3,20})(?![A-Za-z0-9_-])',re.I)
VOC=g['build_vocabulary'](both)
LEAD=g['TITLE_LEAD_RE']; ok_bare=g['_acceptable_bare_name']
# Corpus prose frequency, u/ mentions and urls stripped. TITLE_LEAD_RE currently accepts a
# bare leading name only if it is written with a u/ prefix somewhere; that misses accounts
# nobody ever prefixed (ExampleNameA, examplefly5791). Rarity is the second corroboration:
# category prefixes (Request, Discussion, AMA, Tip) recur throughout the prose, usernames do not.
# --- 3. Extra title rules added after reading posts where accounts were missed (DATA_COLLECTION.md, 'Extraction rules') ---
import collections
COMMA_LEAD=re.compile(r'^\s*([A-Za-z0-9_-]{3,20})\s*,\s+')
# "exname56, exname5456, exname5264. x-posting things from r/pics" - a leading RUN of
# comma-separated names, then prose. The single-name comma rule takes only the first.
LEAD_RUN=re.compile(r'^\s*((?:[A-Za-z0-9_-]{3,20}\s*,\s*)+(?:and\s+)?[A-Za-z0-9_-]{3,20})(?![A-Za-z0-9_-])')
# "ExampleSurname's 8 months old account" - a BARE possessive (no u/ prefix). The notebook's
# possessive rule means the victim, but it only fires on u/Name's; a bare one is the subject.
# The lookbehind must exclude '-' too, or "u/Example-name31's" yields "name31".
POSS=re.compile(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_-]{3,20})['\u2019]s\b")
POSS_NOUN=re.compile(r'^\s*(?:\d+\s+\w+\s+old\s+)?(?:account|profile|alt|bot|post|posts|comment|comments|submission|submissions|history|karma|page|spam)',re.I)
WOKE=re.compile(r'^\s*([A-Za-z0-9_-]{3,20})\s*[,:\u2014\u2013-]*\s*(?:just\s+)?(?:woke|born|was\s+born)',re.I)
# Same username-shape test as username_shape.py, kept here for reference.
SHAPED=re.compile(r'[0-9_-]|[a-z][A-Z]')   # digit, underscore, hyphen or internal capital
TOKEN=re.compile(r'^[A-Za-z0-9_-]{3,20}$')

def username_list_title(title,freq,rare_max):
    """A title that is nothing but bare usernames - 'examplename1 examplename2 examplename3',
    'ExampleName535', 'examplek001, examplek002, examplek003'. No delimiter, no u/ prefix and often
    no link, so every other rule misses them. Accepted only when EVERY token is username-shaped
    and rare in the corpus prose, which is what keeps ordinary prose titles out."""
    toks=[t.strip('.,;:()[]\'"') for t in title.split()]
    toks=[t for t in toks if t]
    if not toks or len(toks)>12: return []
    if not all(looks_like_username(t,freq,rare_max) for t in toks): return []
    return toks
# PROSE strips u/ mentions and URLs, so freq counts how often a word appears as ordinary prose in the corpus.
PROSE=re.compile(r'(?<![A-Za-z0-9_./])/?u/[A-Za-z0-9_-]{3,20}|https?://\S+|www\.\S+')
freq=collections.Counter()
for _t,_b in zip(both.title,both.selftext):
    freq.update(w.lower() for w in re.findall(r'[A-Za-z]{3,20}',PROSE.sub(' ',(_t or '')+' '+(_b or ''))))

# --- 4. Extract accused names post by post; each counter records how many mentions a rule added ---
rec=[]; dropped=0; url_hits=0; url_clash=0; lead_hits=0; comma_hits=0; woke_hits=0; list_hits=0; run_hits=0; poss_hits=0; drop_log=[]
for p in both.itertuples(index=False):
    # Names this post frames as a helper (credited with finding the account) or a victim (copied from) are not accused here.
    roles=non_accused(unesc(p.title))|non_accused(unesc(p.selftext))
    # Base notebook extraction: dict username_key -> (extraction_method, name as written).
    found=ex(p.title,p.selftext,VOC)
    # the submission's own link target, when it is a profile
    # Leading run of comma-separated names; only accepted if every name has username shape and is rare.
    mrun=LEAD_RUN.search(p.title)
    if mrun:
        toks=[t.strip() for t in re.split(r',|\band\b',mrun.group(1)) if t.strip()]
        if len(toks)>=2 and all(looks_like_username(t,freq,RARE_MAX) for t in toks):
            for nm in toks:
                k=nm.lower()
                if k not in RES and k not in found: found[k]=('title_lead_run',nm); run_hits+=1
    # Bare possessive 'Name's account/post...' (no u/ prefix) names the subject, not a victim.
    for mp in POSS.finditer(p.title):
        nm=mp.group(1); k=nm.lower()
        if p.title[max(0,mp.start()-2):mp.start()].lower().endswith('u/'): continue
        if not looks_like_username(nm,freq,RARE_MAX): continue
        if not POSS_NOUN.match(p.title[mp.end():]): continue
        if k not in RES and k not in found: found[k]=('title_possessive',nm); poss_hits+=1
    for nm in username_list_title(p.title,freq,RARE_MAX):
        k=nm.lower()
        if k not in RES and k not in found:
            found[k]=('title_username_list',nm); list_hits+=1
    # comma-delimited lead ("ExampleNameC, another copy/paste bot") - the notebook's lead rule
    # only accepts : - -- as delimiters. Gated on username SHAPE as well as rarity, because
    # "Wao, good, nice, funny: ..." is quoted spam text, not an account.
    mc=COMMA_LEAD.search(p.title)
    if mc:
        nm=mc.group(1); k=nm.lower()
        if ok_bare(nm) and k not in RES and k not in found and looks_like_username(nm,freq,RARE_MAX):
            found[k]=('title_lead_comma',nm); comma_hits+=1
    # "Examplename18, woke up today after 5 years" - dormant-account house style; TITLE_BORN_RE
    # requires the word "born" and a : or - delimiter, so a bare "woke up" report is missed.
    mw=WOKE.search(p.title)
    if mw:
        nm=mw.group(1); k=nm.lower()
        if ok_bare(nm) and k not in RES and k not in found and freq[k]<RARE_MAX:
            found[k]=('title_woke',nm); woke_hits+=1
    # Bare leading name with ':' or '-' delimiter, accepted when the name is rare as prose.
    mlead=LEAD.search(p.title)
    if mlead:
        nm=mlead.group(1); k=nm.lower()
        if ok_bare(nm) and k not in RES and k not in found and freq[k]<RARE_MAX:
            found[k]=('title_lead_rare',nm); lead_hits+=1
    # Link posts: a profile URL in the submission link is the accusation itself.
    for field in (p.url,p.url_overridden_by_dest):
        for nm in PROFILE.findall(field or ''):
            k=nm.lower()
            if k in RES: continue
            if k in roles: url_clash+=1; continue
            if k not in found: found[k]=('submission_url',nm); url_hits+=1
    # Drop names that appear in a helper/victim context in this same post; keep the rest as accusing mentions.
    for key,(method,written) in found.items():
        if key in roles:
            dropped+=1; drop_log.append({'username_key':key,'post_id':p.id}); continue
        rec.append({'username_key':key,'username':written,'extraction_method':method,
                    'post_id':p.id,'created_utc':p.created_utc,'source':p.source})
m=pd.DataFrame(rec)
print(f'{len(m):,} accusing mentions | {m.username_key.nunique():,} accounts',flush=True)
print(f'  added from submission url: {url_hits:,} mentions | url name also in a role context: {url_clash:,}',flush=True)
print(f'  added from rare bare lead: {lead_hits:,} mentions',flush=True)
print(f'  added from comma lead    : {comma_hits:,} mentions',flush=True)
print(f'  added from username-list titles: {list_hits:,} mentions',flush=True)
print(f'  added from comma-separated run : {run_hits:,} mentions',flush=True)
print(f'  added from bare possessive     : {poss_hits:,} mentions',flush=True)
print(f'  added from woke/born lead: {woke_hits:,} mentions',flush=True)
print(f'  role mentions excluded   : {dropped:,}',flush=True)

# --- 5. One row per account: first mention time and post, number of distinct accusing posts, spelling used most often ---
ms=m.sort_values('created_utc')
acc=(ms.groupby('username_key').agg(times_mentioned=('post_id','nunique'),
       first_mention_utc=('created_utc','min'), first_post_id=('post_id','first'),
       extraction_method=('extraction_method','first'), first_source=('source','first')).reset_index())
# Most common capitalisation of each username across mentions.
sp=(m.groupby(['username_key','username']).size().sort_values(ascending=False)
      .reset_index(name='n').drop_duplicates('username_key').set_index('username_key')['username'])
acc['username']=acc.username_key.map(sp)
acc['first_mention_date']=pd.to_datetime(acc.first_mention_utc,unit='s')
acc['sources']=acc.username_key.map(m.groupby('username_key')['source'].agg(lambda s:'both' if s.nunique()>1 else s.iloc[0]))
acc=acc[['username','username_key','first_mention_utc','first_mention_date','times_mentioned',
         'first_post_id','extraction_method','first_source','sources']]
# Sanity checks: unique keys, at least one mention each, valid username length, no missing values.
assert acc.username_key.is_unique and acc.times_mentioned.min()>=1
assert acc.username.str.len().between(3,20).all() and acc.isna().sum().sum()==0
# Compare with the previous table to print what changed, then overwrite it.
prev=set(pd.read_parquet('data/accused_accounts_final.parquet').username_key)
acc.to_parquet('data/accused_accounts_final.parquet',engine='pyarrow',compression='zstd',index=False)
pd.DataFrame(drop_log).to_parquet('data/excluded_role_mentions.parquet',engine='pyarrow',compression='zstd',index=False)
print(f'\nfinal table: {len(acc):,} accounts (was {len(prev):,}) | gained {len(set(acc.username_key)-prev):,} | lost {len(prev-set(acc.username_key)):,}',flush=True)
print(acc.extraction_method.value_counts().to_string(),flush=True)
print('\nby era:'); print(acc.sources.value_counts().to_string(),flush=True)
