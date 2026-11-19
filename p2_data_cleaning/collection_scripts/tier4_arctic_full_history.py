# WHAT: Tier 4. Pulls each non-deleted accused account's post and comment history from Arctic Shift,
#       up to CAP items per stream; accounts over the cap are marked history_truncated.
# READS: data/account_meta_arctic.parquet (Tier 3), data/accused_accounts_not_deleted.parquet, data/account_status_checkpoint_full.parquet.
# WRITES: data/arctic_posts/part-*.parquet, data/arctic_comments/part-*.parquet, data/arctic_pull_done.parquet (ledger, one row per account).
# RUN: python tier4_arctic_full_history.py [limit]. Resumable: accounts already in the ledger are skipped.
"""Tier 4 - full Arctic Shift post+comment history. Resumable, 5 workers @ 5 req/s.
Accounts are processed cheapest-first so partial completion still covers the most accounts."""
import os, sys, time, threading, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from requests_ratelimiter import LimiterSession

os.chdir('/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning')
META='data/account_meta_arctic.parquet'
TABLE='data/accused_accounts_not_deleted.parquet'
STATUS='data/account_status_checkpoint_full.parquet'
LEDGER='data/arctic_pull_done.parquet'
POSTDIR='data/arctic_posts'; COMDIR='data/arctic_comments'
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else None
FLUSH=250
CAP=10000   # max items pulled per stream (posts, comments) per account
B='https://arctic-shift.photon-reddit.com/api'

# Fields kept for posts and comments.
POST_COLS=['id','name','created_utc','author','author_created_utc','author_fullname','subreddit',
 'subreddit_id','subreddit_subscribers','title','selftext','url','domain','url_overridden_by_dest',
 'is_self','is_video','is_original_content','is_reddit_media_domain','score','ups','upvote_ratio',
 'num_comments','num_crossposts','total_awards_received','gilded','over_18','locked','stickied',
 'removed_by_category','quarantine','retrieved_utc','permalink','distinguished','edited','link_flair_text']
COM_COLS=['id','link_id','parent_id','body','created_utc','author','subreddit','subreddit_id','score',
 'ups','downs','controversiality','edited','gilded','distinguished','removal_reason','permalink','retrieved_on']
# Fields whose type varies between items; stored as strings so part files share one schema.
STR_COLS={'edited','distinguished','removal_reason','removed_by_category','link_flair_text','author_fullname'}

os.makedirs(POSTDIR,exist_ok=True); os.makedirs(COMDIR,exist_ok=True)
# Accounts to pull: non-deleted, found by Tier 3, with at least one item.
meta=pd.read_parquet(META)
meta=meta[meta.username_key.isin(set(pd.read_parquet(TABLE).username_key))]   # non-deleted only
meta=meta[meta.found.astype(bool)].copy()
meta['n_posts']=pd.to_numeric(meta.num_posts,errors='coerce').fillna(0).astype(int)
meta['n_comments']=pd.to_numeric(meta.num_comments,errors='coerce').fillna(0).astype(int)
meta['items']=meta.n_posts+meta.n_comments
meta=meta[meta['items']>0]
# Also skip accounts Tier 2 already found deleted or forbidden.
if os.path.exists(STATUS):
    st=pd.read_parquet(STATUS)
    dead=set(st.loc[st.account_status.isin(['deleted','forbidden']),'username_key'])
    before=len(meta); meta=meta[~meta.username_key.isin(dead)]
    print(f'excluded {before-len(meta):,} accounts already known deleted/forbidden',flush=True)
ledger=pd.read_parquet(LEDGER).to_dict('records') if os.path.exists(LEDGER) else []
for r in ledger:                      # rows collected before the cap were pulled in full
    r.setdefault('history_truncated',False); r.setdefault('cap',None)
done={r['username_key'] for r in ledger}
# Smallest accounts first.
meta=meta[~meta.username_key.isin(done)].sort_values('items')
if LIMIT: meta=meta.head(LIMIT)
print(f'resuming: {len(done):,} accounts already pulled',flush=True)
print(f'{len(meta):,} accounts this pass | {meta["items"].sum():,} items | '
      f'~{(np.ceil(meta.n_posts/100)+np.ceil(meta.n_comments/100)).sum():,.0f} requests',flush=True)

session=LimiterSession(per_second=5)
session.headers['User-Agent']='dissertation-aft-collection/1.0 (research)'
lock=threading.Lock(); pbuf=[]; cbuf=[]; t0=time.time(); n_done=0
# Number new part files after the ones already on disk.
part=len([f for f in os.listdir(POSTDIR) if f.endswith('.parquet')])

# Keep only the listed fields, cast the mixed-type ones to string, tag each row with the account key.
def norm(recs,cols,key):
    out=[]
    for d in recs:
        r={c:d.get(c) for c in cols}
        for c in STR_COLS&set(cols):
            if r[c] is not None: r[c]=str(r[c])
        r['outer_key']=key
        out.append(r)
    return out

def pull(kind,user,cap=CAP):
    """Arctic Shift treats `after` as EXCLUSIVE, so `after = last_created_utc` silently drops
    every remaining item sharing that timestamp - a real loss on high-rate accounts (verified:
    11,800 of 11,857 comments for u/example-acct). Step back one second so pages OVERLAP instead of
    skipping, and dedupe by id. Costs one duplicate page row per boundary; loses nothing."""
    seen={}; after=None; pages=0
    while True:
        p={'author':user,'limit':100,'sort':'asc'}
        if after is not None: p['after']=after
        data=[]; retries=0; ok=False
        while retries<10:
            try:
                r=session.get(f'{B}/{kind}/search',params=p,timeout=40)
                if r.status_code==200:
                    data=r.json().get('data') or []; ok=True; break
                retries+=1; time.sleep(min(30,2*retries))
            except Exception:
                retries+=1; time.sleep(min(30,2*retries))
        if not ok: raise RuntimeError(f'{kind} pagination failed for {user}')
        if not data: break
        new=[x for x in data if x['id'] not in seen]
        for x in data: seen.setdefault(x['id'],x)
        # Stop on "no NEW ids", not on "short page". A short page is not a reliable
        # end-of-stream signal: a transient one mid-stream would silently truncate.
        # (Measured 2026-10-05 on 20 accounts: the API does not emit spurious short
        # pages, so this recovers nothing today - it is insurance, not a repair.)
        if not new:
            if len(data)<100: break
            after=max(x['created_utc'] for x in data); continue   # whole page already held
        if len(seen)>=cap: break
        nxt=max(x['created_utc'] for x in data)-1
        if after is not None and nxt<=after: nxt=after+1   # guarantee forward progress
        after=nxt
        pages+=1
        if pages>5000: break                                # runaway guard
    return list(seen.values())[:cap]

# Write buffered rows as new part files and save the ledger.
def flush():
    global part,pbuf,cbuf
    if pbuf:
        pd.DataFrame(pbuf).to_parquet(f'{POSTDIR}/part-{part:05d}.parquet',engine='pyarrow',compression='zstd',index=False)
    if cbuf:
        pd.DataFrame(cbuf).to_parquet(f'{COMDIR}/part-{part:05d}.parquet',engine='pyarrow',compression='zstd',index=False)
    pd.DataFrame(ledger).to_parquet(LEDGER,engine='pyarrow',compression='zstd',index=False)
    pbuf=[]; cbuf=[]; part+=1

# Pull one account. complete = got every expected item (or the cap); errors are recorded, not raised.
def work(row):
    global n_done
    u=row.username_key
    rec={'username_key':u,'expected_posts':row.n_posts,'expected_comments':row.n_comments,
         'n_posts':0,'n_comments':0,'complete':False,'error':None,
         'history_truncated':bool(row.n_posts>CAP or row.n_comments>CAP),'cap':CAP}
    try:
        ps=pull('posts',u) if row.n_posts>0 else []
        cs=pull('comments',u) if row.n_comments>0 else []
        rec['n_posts']=len(ps); rec['n_comments']=len(cs)
        rec['complete']=(len(ps)>=min(row.n_posts,CAP) and len(cs)>=min(row.n_comments,CAP))
        P=norm(ps,POST_COLS,u); C=norm(cs,COM_COLS,u)
    except Exception as e:
        rec['error']=f'{type(e).__name__}: {e}'; P=[]; C=[]
    # Buffers and ledger are shared between threads; flush every FLUSH accounts.
    with lock:
        pbuf.extend(P); cbuf.extend(C); ledger.append(rec); n_done+=1
        if n_done%FLUSH==0 or n_done==len(meta):
            flush(); el=time.time()-t0
            print(f'  {n_done:,}/{len(meta):,} accounts | {n_done/el*3600:,.0f}/h | '
                  f'~{(len(meta)-n_done)/(n_done/el)/3600:.1f} h left',flush=True)

with ThreadPoolExecutor(max_workers=5) as ex:
    list(ex.map(work,[r for r in meta.itertuples(index=False)]))
with lock: flush()
L=pd.DataFrame(ledger)
print(f'\nTIER 4 PASS COMPLETE: {len(L):,} accounts in ledger',flush=True)
print(f'  complete          : {L.complete.sum():,} ({L.complete.mean():.1%})',flush=True)
print(f'  errors            : {L.error.notna().sum():,}',flush=True)
print(f'  posts pulled      : {L.n_posts.sum():,} (expected {L.expected_posts.sum():,})',flush=True)
print(f'  comments pulled   : {L.n_comments.sum():,} (expected {L.expected_comments.sum():,})',flush=True)
print(f'  history_truncated : {L.history_truncated.sum():,} accounts capped at {CAP:,}/stream',flush=True)
