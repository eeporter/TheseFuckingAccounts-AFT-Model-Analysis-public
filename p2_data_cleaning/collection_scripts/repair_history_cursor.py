# WHAT: re-pulls the full post and comment history of every account whose pull needed more than one page,
#       with the fixed cursor, and appends the rows beside the originals.
# READS: data/arctic_pull_done.parquet (ledger), data/cursor_at_risk.parquet (if present), data/cursor_repair_ledger.parquet (resume).
# WRITES: data/arctic_posts/repair-*.parquet, data/arctic_comments/repair-*.parquet, data/cursor_repair_ledger.parquet.
# RUN: python repair_history_cursor.py (resumable: accounts already in the repair ledger are skipped).
"""Targeted cursor repair. Fixed cursor (after = ts-1 + id dedupe), union-append semantics:
repaired rows are ADDED alongside the originals, never overwriting them, so a short re-pull
can never destroy data. Canonical read is .drop_duplicates(['outer_key','id'])."""
import os, time, threading, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from requests_ratelimiter import LimiterSession

os.chdir('/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning')
LEDGER='data/arctic_pull_done.parquet'; RISK='data/cursor_at_risk.parquet'
OUT='data/cursor_repair_ledger.parquet'
POSTDIR='data/arctic_posts'; COMDIR='data/arctic_comments'
B='https://arctic-shift.photon-reddit.com/api'; FLUSH=100
CAP=10000   # same per-stream cap as the capped Tier 4 run

# Fields kept for posts and comments (same as the Tier 4 pull).
POST_COLS=['id','name','created_utc','author','author_created_utc','author_fullname','subreddit',
 'subreddit_id','subreddit_subscribers','title','selftext','url','domain','url_overridden_by_dest',
 'is_self','is_video','is_original_content','is_reddit_media_domain','score','ups','upvote_ratio',
 'num_comments','num_crossposts','total_awards_received','gilded','over_18','locked','stickied',
 'removed_by_category','quarantine','retrieved_utc','permalink','distinguished','edited','link_flair_text']
COM_COLS=['id','link_id','parent_id','body','created_utc','author','subreddit','subreddit_id','score',
 'ups','downs','controversiality','edited','gilded','distinguished','removal_reason','permalink','retrieved_on']
# Fields whose type varies between items; stored as strings so part files share one schema.
STR_COLS={'edited','distinguished','removal_reason','removed_by_category','link_flair_text','author_fullname'}

led=pd.read_parquet(LEDGER)
if 'history_truncated' not in led: led['history_truncated']=False
led['history_truncated']=led.history_truncated.fillna(False).astype(bool)
# EVERY account whose stream exceeded one page used the cursor and is therefore exposed.
# Accounts with <=100 items per stream never paginated and are provably unaffected.
paginated=led[(led.n_posts>100)|(led.n_comments>100)]
targets=sorted(paginated.username_key)
# Counted for the log only: accounts the ledger already shows as short, and accounts flagged at risk.
short=set(led.loc[(~led.complete.astype(bool)) & (~led.history_truncated),'username_key'])
risk=set(pd.read_parquet(RISK).username_key) if os.path.exists(RISK) else set()
# Resume: skip accounts already in the repair ledger.
prev=pd.read_parquet(OUT).to_dict('records') if os.path.exists(OUT) else []
done={r['username_key'] for r in prev}
targets=[t for t in targets if t not in done]
info=led.set_index('username_key')
print(f'targets: {len(targets):,} paginated accounts (of {len(led):,} total; {len(led)-len(paginated):,} never paginated = provably unaffected)',flush=True)
print(f'  includes {len(short):,} observed-short and {len(risk):,} statically at-risk',flush=True)
print(f'  already repaired: {len(done):,}',flush=True)

session=LimiterSession(per_second=5)
session.headers['User-Agent']='dissertation-aft-collection/1.0 (research)'
lock=threading.Lock(); rows=list(prev); pbuf=[]; cbuf=[]; n=0; t0=time.time()
# Number the new part files after any repair files already written.
part=len([f for f in os.listdir(POSTDIR) if f.startswith('repair-')])

# Page through one account's posts or comments, oldest first, up to the cap.
def pull(kind,user,cap=CAP):
    seen={}; after=None; pages=0
    while True:
        p={'author':user,'limit':100,'sort':'asc'}
        if after is not None: p['after']=after
        data=[]; ok=False
        # Up to 10 tries per page with growing waits.
        for att in range(10):
            try:
                r=session.get(f'{B}/{kind}/search',params=p,timeout=40)
                if r.status_code==200: data=r.json().get('data') or []; ok=True; break
                time.sleep(min(30,2*(att+1)))
            except Exception: time.sleep(min(30,2*(att+1)))
        if not ok: raise RuntimeError(f'{kind} failed for {user}')
        if not data: break
        new=[x for x in data if x['id'] not in seen]
        for x in data: seen.setdefault(x['id'],x)
        # Stop on "no NEW ids", not on "short page" - a transient short page mid-stream
        # would otherwise silently truncate the account.
        if not new:
            if len(data)<100: break
            after=max(x['created_utc'] for x in data); continue
        if len(seen)>=cap: break
        nxt=max(x['created_utc'] for x in data)-1  # overlap instead of skip
        if after is not None and nxt<=after: nxt=after+1
        after=nxt
        pages+=1
        if pages>5000: break
    return list(seen.values())

# Keep only the listed fields, cast the mixed-type ones to string, and tag each row with the account key.
def norm(recs,cols,key):
    out=[]
    for d in recs:
        r={c:d.get(c) for c in cols}
        for c in STR_COLS&set(cols):
            if r[c] is not None: r[c]=str(r[c])
        r['outer_key']=key; out.append(r)
    return out

# Write the buffered rows as new part files and save the repair ledger.
def flush():
    global part,pbuf,cbuf
    if pbuf: pd.DataFrame(pbuf).to_parquet(f'{POSTDIR}/repair-{part:05d}.parquet',engine='pyarrow',compression='zstd',index=False)
    if cbuf: pd.DataFrame(cbuf).to_parquet(f'{COMDIR}/repair-{part:05d}.parquet',engine='pyarrow',compression='zstd',index=False)
    pd.DataFrame(rows).to_parquet(OUT,engine='pyarrow',compression='zstd',index=False)
    pbuf=[]; cbuf=[]; part+=1

# Re-pull one account; record stored vs re-pulled counts, or the error.
def work(u):
    global n
    row=info.loc[u]
    rec={'username_key':u,'stored_posts':int(row.n_posts),'stored_comments':int(row.n_comments),
         'repull_posts':0,'repull_comments':0,'recovered_posts':0,'recovered_comments':0,
         'cursor_repaired':True,'error':None}
    try:
        ps=pull('posts',u) if row.n_posts>0 else []
        cs=pull('comments',u) if row.n_comments>0 else []
        rec['repull_posts']=len(ps); rec['repull_comments']=len(cs)
        rec['recovered_posts']=max(0,len(ps)-int(row.n_posts))
        rec['recovered_comments']=max(0,len(cs)-int(row.n_comments))
        P=norm(ps,POST_COLS,u); C=norm(cs,COM_COLS,u)
    except Exception as e:
        rec['error']=f'{type(e).__name__}: {e}'; rec['cursor_repaired']=False; P=[]; C=[]
    # Buffers are shared between threads; flush every FLUSH accounts.
    with lock:
        pbuf.extend(P); cbuf.extend(C); rows.append(rec); n+=1
        if n%FLUSH==0 or n==len(targets):
            flush(); el=time.time()-t0
            print(f'  {n:,}/{len(targets):,} | {n/el*3600:,.0f}/h | ~{(len(targets)-n)/(n/el)/60:.0f} min left',flush=True)

# 5 parallel workers sharing the 5 requests/second limit.
with ThreadPoolExecutor(max_workers=5) as ex: list(ex.map(work,targets))
with lock: flush()
R=pd.DataFrame(rows)
print(f'\nREPAIR COMPLETE: {len(R):,} accounts',flush=True)
print(f'  errors            : {R.error.notna().sum():,}',flush=True)
print(f'  accounts recovering data: {(R.recovered_posts+R.recovered_comments>0).sum():,}',flush=True)
print(f'  items recovered   : {R.recovered_posts.sum():,} posts + {R.recovered_comments.sum():,} comments',flush=True)
print(f'  re-pull returned FEWER (stored kept by union): {((R.repull_posts<R.stored_posts)|(R.repull_comments<R.stored_comments)).sum():,}',flush=True)
