"""Busiest-subreddit thread counts per 7 days, used to choose the flooding cut-off (2026-10-15; set at 10+ threads).

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../aft_model/); kept for the record.
For each account and kind ('threads' = posts + comments by thread, 'posts' = own posts only), tracks over follow-up the
number of active threads in the account's busiest subreddit (the numerator of flooding) and summarises it, with medians
weighted by how long each value lasts.
Inputs: the setup of build_exact.py (exec'd: data, censoring, thread_active). Output: busiest.parquet next to this
script and printed percentiles. Run: python busiest.py."""
import os, numpy as np, pandas as pd
D=os.path.dirname(os.path.abspath(__file__))  # busiest threads per 7 days in one subreddit (flooding cut-off, 2026-10-15)
# exec build_exact.py up to its episode loop; __file__ is swapped so its HERE points at this folder's build.py
src=open(os.path.join(D,'build_exact.py')).read().split('t0c = time.time()')[0]
_me=__file__; __file__=os.path.join(D,'build_exact.py'); exec(src); __file__=_me
def wmed(v,w):
    """Weighted median of values v with weights w (arrays); returns the lower weighted median."""
    o=np.argsort(v); v,w=v[o],w[o]; c=np.cumsum(w); return v[np.searchsorted(c,c[-1]/2)]
R=[]
pg,cg=posts.groupby('outer_key'),coms.groupby('outer_key')
for k,a in acc.iterrows():
    end=min(a.days_to_accused,cens.get(k,np.inf))
    p=pg.get_group(k) if k in pg.groups else posts.iloc[:0]; c=cg.get_group(k) if k in cg.groups else coms.iloc[:0]
    p,c=p[p.t<end],c[c.t<end]
    for kind,act in (('threads',pd.concat([p[['thread','t','subreddit']],c[['thread','t','subreddit']]])),('posts',p[['thread','t','subreddit']])):
        if act.empty: R.append(dict(k=k,kind=kind,ever=0)); continue
        # same change points as build_exact.py; dur = how long each grid value lasts
        pts=np.r_[0.0,act.t,act.t+WIN,act.groupby('thread').t.max().to_numpy()+WIN]
        grid=np.unique(pts[(pts>=0)&(pts<end)]); dur=np.diff(np.r_[grid,end])
        tot=np.zeros(len(grid)); mx=np.zeros(len(grid))
        for _,s in act.groupby('subreddit'):
            w=thread_active(s,grid); tot+=w; mx=np.maximum(mx,w)
        on=tot>0
        # max: peak busiest-subreddit count; med_active: time-weighted median while any thread is active; med_all: over
        # all follow-up; share_active / share_ge2: share of follow-up with 1+ / 2+ active threads
        R.append(dict(k=k,kind=kind,ever=1,max=mx.max(),med_active=wmed(mx[on],dur[on]) if on.any() and dur[on].sum()>0 else np.nan,
                      med_all=wmed(mx,dur) if dur.sum()>0 else 0, share_active=dur[on].sum()/max(dur.sum(),1e-9),
                      share_ge2=dur[tot>=2].sum()/max(dur.sum(),1e-9)))
r=pd.DataFrame(R); r.to_parquet(os.path.join(D,'busiest.parquet'))
for kind,g in r.groupby('kind'):
    g=g[g.ever==1]; print('\n==',kind,'accounts with any:',len(g))
    for col in ['max','med_active','med_all','share_active']:
        print(col, g[col].describe(percentiles=[.1,.25,.5,.75,.9]).round(2).to_dict())
    print('accounts whose weekly max ever >=2:', (g['max']>=2).sum(), '>=5:', (g['max']>=5).sum(), '>=20:',(g['max']>=20).sum())
