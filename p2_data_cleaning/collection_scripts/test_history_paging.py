# WHAT: test of Arctic Shift paging. For the 20 uncapped accounts missing the most posts, pulls their posts twice:
#       'legacy' (stop at the first page under 100 items) and 'robust' (stop only when a page brings no new ids),
#       and prints how many items the robust stop recovers. This test led to the cursor repair (repair_history_cursor.py).
# READS: data/history_summary.parquet. WRITES: nothing (prints a table).
# RUN: python test_history_paging.py
import pandas as pd, time, threading
from concurrent.futures import ThreadPoolExecutor
from requests_ratelimiter import LimiterSession
import os; os.chdir('/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning')
# Uncapped accounts with fewer stored posts than Arctic Shift's expected count, 150-4,000 posts; take the 20 most short.
hs=pd.read_parquet('data/history_summary.parquet')
u=hs[(~hs.history_truncated)&(hs.n_posts<hs.expected_posts)&(hs.n_posts.between(150,4000))].copy()
u['short']=u.expected_posts-u.n_posts
u=u.nlargest(20,'short')
# Arctic Shift allows about 5 requests a second.
SS=LimiterSession(per_second=5); SS.headers['User-Agent']='dissertation-aft-collection/1.0 (research)'
B='https://arctic-shift.photon-reddit.com/api'
# One page (up to 100 items, oldest first) after timestamp 'after'; 6 tries, then fail.
def fetch(kind,user,after):
    p={'author':user,'limit':100,'sort':'asc'}
    if after is not None: p['after']=after
    for a in range(6):
        try:
            r=SS.get(f'{B}/{kind}/search',params=p,timeout=40)
            if r.status_code==200: return r.json().get('data') or []
        except Exception: pass
        time.sleep(2)
    raise RuntimeError('fail')
# Page through one account's stream; returns (number of distinct items, page sizes).
def pull(kind,user,legacy):
    seen={}; after=None; sizes=[]
    while True:
        d=fetch(kind,user,after); sizes.append(len(d))
        if not d: break
        new=[x for x in d if x['id'] not in seen]
        for x in d: seen.setdefault(x['id'],x)
        # Legacy rule: a page under 100 items ends the stream (a transient short page truncates it).
        if legacy:
            if len(d)<100: break
        else:
            # Robust rule: end only when a page has no new ids; if a full page repeats, move past its timestamp.
            if not new:
                if len(d)<100: break
                after=max(x['created_utc'] for x in d); continue
        # Next cursor = last timestamp - 1, so items sharing the boundary second are not skipped.
        nxt=max(x['created_utc'] for x in d)-1
        if after is not None and nxt<=after: nxt=after+1
        after=nxt
        # Safety stop after 600 pages.
        if len(sizes)>600: break
    return len(seen),sizes
out=[]; lk=threading.Lock()
# Pull one account both ways; keep the short pages seen mid-stream in the robust pull.
def work(r):
    a,_=pull('posts',r.username_key,True)
    b,sb=pull('posts',r.username_key,False)
    with lk: out.append((r.username_key,int(r.expected_posts),int(r.n_posts),a,b,[s for s in sb[:-1] if s<100]))
with ThreadPoolExecutor(max_workers=5) as ex: list(ex.map(work,[r for r in u.itertuples()]))
print(f'{"account":<24}{"meta":>8}{"stored":>8}{"legacy":>8}{"robust":>8}   mid-stream short pages')
rec=0
# Accounts sorted by how many posts were missing.
for n,e,s,a,b,mid in sorted(out,key=lambda x:-(x[1]-x[2])):
    rec+=max(0,b-a)
    print(f'{n:<24}{e:>8}{s:>8}{a:>8}{b:>8}   {len(mid)} {mid[:5]}')
print(f'\nrobust stop recovers over legacy: {rec} items across {len(out)} accounts')
