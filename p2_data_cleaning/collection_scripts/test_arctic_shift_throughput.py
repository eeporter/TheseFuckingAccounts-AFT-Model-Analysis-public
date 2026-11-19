# WHAT: speed test of the Arctic Shift users/search endpoint with 5 vs 8 parallel workers at 5 requests/second,
#       before choosing the worker count for Tier 3/4. READS: data/accused_accounts_all.parquet. WRITES: nothing.
# RUN: python test_arctic_shift_throughput.py from p2_data_cleaning/.
import time, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from requests_ratelimiter import LimiterSession
acc=pd.read_parquet('data/accused_accounts_all.parquet')
# 200 accounts as test queries.
names=list(acc.username_key[200:400])
URL='https://arctic-shift.photon-reddit.com/api/users/search'
for workers in (5,8):
    s=LimiterSession(per_second=5); s.headers['User-Agent']='dissertation-aft-collection/1.0 (research)'
    codes=[]
    t0=time.time()
    # One lookup: HTTP status and number of rows returned (-1 on an error status).
    def one(n):
        r=s.get(URL,params={'author':n},timeout=30)
        return r.status_code, len(r.json().get('data') or []) if r.status_code==200 else -1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sc,n in ex.map(one,names): codes.append((sc,n))
    el=time.time()-t0
    # Requests per second, error statuses, and lookups that returned more than one row.
    ok=sum(1 for c,_ in codes if c==200); bad=[c for c,_ in codes if c!=200]
    multi=sum(1 for _,n in codes if n>1)
    print(f'workers={workers}: {len(names)} reqs in {el:.1f}s = {len(names)/el:.1f}/s | 200s={ok} | non-200={sorted(set(bad))} | rows>1={multi}')
