# WHAT: Tier 3. For every accused account that is not deleted, asks Arctic Shift's users/search for its _meta
#       (first/last post and comment times, post and comment counts, karma). The counts set the Tier 4 budget.
# READS: data/accused_accounts_not_deleted.parquet. WRITES: data/account_meta_arctic.parquet (rewritten every 500 accounts).
# RUN: python tier3_arctic_account_meta.py [limit]. Resumable.
"""Tier 3 - Arctic Shift users/search _meta for every accused account. Resumable, 5 workers @ 5 req/s."""
import os, sys, time, threading, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from requests_ratelimiter import LimiterSession

os.chdir('/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning')
ACCOUNTS='data/accused_accounts_not_deleted.parquet'
OUT='data/account_meta_arctic.parquet'
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else None
EVERY=500
URL='https://arctic-shift.photon-reddit.com/api/users/search'
# Fields kept from Arctic Shift's _meta block.
META_COLS=['earliest_post_at','earliest_comment_at','last_post_at','last_comment_at',
           'num_posts','num_comments','post_karma','comment_karma','total_karma',
           'post_stats_updated_at','comment_stats_updated_at']

accounts=pd.read_parquet(ACCOUNTS)
# Resume: accounts already in the output are skipped.
rows=pd.read_parquet(OUT).to_dict('records') if os.path.exists(OUT) else []
done={r['username_key'] for r in rows}
print(f'resuming: {len(done):,} already fetched',flush=True)
todo=[u for u in accounts.username_key if u not in done]
if LIMIT: todo=todo[:LIMIT]
print(f'{len(todo):,} to fetch this pass ({len(accounts)-len(done):,} outstanding overall)',flush=True)

session=LimiterSession(per_second=5)
session.headers['User-Agent']='dissertation-aft-collection/1.0 (research)'
lock=threading.Lock(); t0=time.time(); n_done=0

def save():
    pd.DataFrame(rows).to_parquet(OUT,engine='pyarrow',compression='zstd',index=False)

# Look up one account; up to 10 tries. found = False when Arctic Shift has no record of it.
def fetch(name):
    global n_done
    rec={'username_key':name,'found':False,'arctic_author':None,'arctic_id':None,'fetch_error':None}
    for c in META_COLS: rec[c]=None
    retries=0
    while retries<10:
        try:
            r=session.get(URL,params={'author':name},timeout=30)
            if r.status_code==200:
                data=r.json().get('data') or []
                # A username should match at most one archive record; more would mean a lookup bug, so stop.
                assert len(data)<=1, f'{name}: expected <=1 row, got {len(data)}'
                if data:
                    d=data[0]; m=d.get('_meta') or {}
                    rec.update({'found':True,'arctic_author':d.get('author'),'arctic_id':d.get('id')})
                    for c in META_COLS: rec[c]=m.get(c)
                break
            retries+=1; time.sleep(min(30,2*retries))
        except AssertionError: raise
        except Exception as e:
            retries+=1; rec['fetch_error']=type(e).__name__; time.sleep(min(30,2*retries))
    if retries>=10 and not rec['found']: rec['fetch_error']=rec['fetch_error'] or 'retries_exhausted'
    with lock:
        rows.append(rec); n_done+=1
        if n_done%EVERY==0 or n_done==len(todo):
            save(); el=time.time()-t0; rate=n_done/el
            print(f'  {n_done:,}/{len(todo):,} | {rate:.1f}/s | {len(rows):,} total | ~{(len(todo)-n_done)/rate/3600:.1f} h left',flush=True)

with ThreadPoolExecutor(max_workers=5) as ex:
    list(ex.map(fetch,todo))
save()

# --- Summary: archive coverage and the number of requests a full Tier 4 pull would need (100 items per request) ---
df=pd.DataFrame(rows)
np_=pd.to_numeric(df.num_posts,errors='coerce').fillna(0)
nc_=pd.to_numeric(df.num_comments,errors='coerce').fillna(0)
items=np_+nc_
pages=(np.ceil(np_/100)+np.ceil(nc_/100))
pages=pages.where(~df.found.astype(bool),pages.clip(lower=2))
print(f'\nTIER 3 COMPLETE: {len(df):,} accounts',flush=True)
print(f'  found in archive : {df.found.sum():,} ({df.found.mean():.1%})',flush=True)
print(f'  fetch errors     : {df.fetch_error.notna().sum():,}',flush=True)
print(f'  items: total {items.sum():,.0f} | median {items.median():.0f} | p90 {items.quantile(.9):,.0f} | max {items.max():,.0f}',flush=True)
print(f'  TIER 4 BUDGET    : {pages.sum():,.0f} requests = {pages.sum()/4.8/3600:.1f} h at 4.8/s',flush=True)
# Request budget if each stream were capped at 500, 1,000 or 5,000 items.
for cap in (500,1000,5000):
    pc=(np.ceil(np_.clip(upper=cap)/100)+np.ceil(nc_.clip(upper=cap)/100)).where(~df.found.astype(bool),0).clip(lower=0)
    pc=(np.ceil(np_.clip(upper=cap)/100)+np.ceil(nc_.clip(upper=cap)/100))
    print(f'    if capped at {cap:,} items/account: {pc.sum():,.0f} requests = {pc.sum()/4.8/3600:.1f} h',flush=True)
