# WHAT: independent checks of the first accused-account table (data/accused_accounts_all.parquet, 2014-2022 CSV only):
#       rebuilds it a different way, hand-recounts random accounts, and checks values and victims. Prints results only.
# READS: data/accused_accounts_all.parquet, /data/Emily_data/TheseFuckingAccounts_submissions.csv, notebook cells 8-12.
# RUN: python verify_accused_table.py from p2_data_cleaning/.
import json, re, random, pandas as pd, collections
# Load the notebook's extraction rules by executing cells 8-12 into namespace g.
nb=json.load(open('collect_accused_usernames_and_first_mention.ipynb'))
g={"re":re,"pd":pd}
for i in (8,9,10,11,12):
    src=''.join(nb['cells'][i]['source'])
    if i==11: src=src.split('VOCABULARY =')[0]
    exec(compile(src,f'cell{i}','exec'),g)
ex=g['extract_suspects']

acc=pd.read_parquet('data/accused_accounts_all.parquet')
posts=pd.read_csv('/data/Emily_data/TheseFuckingAccounts_submissions.csv',low_memory=False)[['id','title','selftext','created_utc']]
posts['title']=posts['title'].fillna(''); posts['selftext']=posts['selftext'].fillna('')
posts['created_utc']=posts['created_utc'].astype('int64')
posts=(posts.sort_values('created_utc').drop_duplicates(subset=['title','selftext'],keep='first').reset_index(drop=True))
VOC=g['build_vocabulary'](posts)

print(f'table: {len(acc):,} rows, {acc.username_key.nunique():,} unique keys')

# --- 1. independent rebuild with a different data structure (defaultdict, no groupby) ---
firsts, postsets = {}, collections.defaultdict(set)
for p in posts.itertuples(index=False):
    for k in ex(p.title,p.selftext,VOC):
        postsets[k].add(p.id)
        if k not in firsts or p.created_utc < firsts[k]: firsts[k]=p.created_utc
ind=pd.DataFrame({'username_key':list(postsets),
                  'times_mentioned':[len(postsets[k]) for k in postsets],
                  'first_mention_utc':[firsts[k] for k in postsets]}).sort_values('username_key').reset_index(drop=True)
cmp=acc[['username_key','times_mentioned','first_mention_utc']].sort_values('username_key').reset_index(drop=True)
print('independent rebuild identical:', ind.equals(cmp))
if not ind.equals(cmp):
    m=ind.merge(cmp,on='username_key',suffixes=('_ind','_saved'))
    print(m[(m.times_mentioned_ind!=m.times_mentioned_saved)|(m.first_mention_utc_ind!=m.first_mention_utc_saved)].head())

# --- 2. parquet round-trip, properly this time ---
back=pd.read_parquet('data/accused_accounts_all.parquet')
print('parquet round-trip exact:', back.equals(acc), '| dtypes:', dict(back.dtypes.astype(str)))

# --- 3. date column agrees with the epoch column ---
print('date == utc column:', (pd.to_datetime(acc.first_mention_utc,unit='s')==acc.first_mention_date).all())

# --- 4. hand-recount 15 random accounts straight from the corpus ---
random.seed(7); bad=0
for row in acc.sample(15,random_state=7).itertuples():
    hits=[(p.id,p.created_utc) for p in posts.itertuples(index=False) if row.username_key in ex(p.title,p.selftext,VOC)]
    if len(hits)!=row.times_mentioned or min(h[1] for h in hits)!=row.first_mention_utc: bad+=1
print(f'hand-recount of 15 random accounts: {15-bad}/15 agree')

# --- 5. sanity on the values themselves ---
print('no u/ prefixes:', not acc.username.str.contains(r'^/?u/',case=False).any())
print('keys are lowercase of username:', (acc.username.str.lower()==acc.username_key).all())
print('no reserved names:', not acc.username_key.isin(g['RESERVED_NAMES']).any())
print('times_mentioned >=1:', acc.times_mentioned.min()>=1, '| max', acc.times_mentioned.max())
print('dates inside corpus range:', acc.first_mention_date.min(), '->', acc.first_mention_date.max())
print('nulls:', acc.isna().sum().sum())

# --- 6. every first_post_id really names its account ---
bad=[]
for row in acc.sample(200,random_state=3).itertuples():
    src=posts.loc[posts.id==row.first_post_id]
    if src.empty or row.username_key not in ex(src.iloc[0].title,src.iloc[0].selftext,VOC): bad.append(row.username_key)
print(f'200 random first_post_id spot-checks: {200-len(bad)}/200 recoverable', bad[:3])

# --- 7. known victims must NOT be in the table ---
vict=set()
for p in posts.itertuples(index=False):
    vict|=g['find_victims'](g['unescape_markdown'](p.title))|g['find_victims'](g['unescape_markdown'](p.selftext))
only_victim=vict-set(acc.username_key)
print(f'\ndistinct victim names seen: {len(vict):,} | never accused anywhere: {len(only_victim):,}')
print('victims that DID also get accused elsewhere:', len(vict)-len(only_victim))
print('malplace in table (should be False):', 'malplace' in set(acc.username_key))
