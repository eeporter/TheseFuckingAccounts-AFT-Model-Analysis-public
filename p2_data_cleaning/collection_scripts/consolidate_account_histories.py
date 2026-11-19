# WHAT: turns the raw Arctic Shift pull folders into the clean history dataset used downstream.
# READS: data/accused_accounts_not_deleted.parquet, data/arctic_posts/, data/arctic_comments/, data/arctic_pull_done.parquet.
# WRITES: data/history_posts/ and data/history_comments/ (emptied and rewritten), data/history_summary.parquet.
# RUN: python consolidate_account_histories.py
"""Build the canonical history dataset.

Fixes three defects in the raw pull directories:
  1. 41% duplicate rows - the cursor repair APPENDED re-pulled rows beside the originals
     (deliberately, so a short re-pull could not destroy data). Deduped by (outer_key,id).
  2. Rows for accounts that turned out deleted, or were later excluded as helpers/victims -
     Tier 4 started before Tier 2 finished, so it could not know. Filtered to the final
     non-deleted table.
  3. Per-account counts in arctic_pull_done.parquet are PRE-repair for 434 accounts.
     Recomputed from what is actually on disk.
"""
import os, gc, glob, collections, pandas as pd, numpy as np
import pyarrow as pa, pyarrow.parquet as pq

def unified_schema(files):
    """Part files disagree on column TYPES: a batch where a column was entirely None got
    typed `null`, and int64 vs double drifted with missing values. pyarrow cannot unify
    null with string, so reading the directory raises ArrowNotImplementedError. Build one
    schema per column: a real type beats null, and double beats int64 (int64 casts up)."""
    seen=collections.defaultdict(list)
    for f in files:
        for fld in pq.read_schema(f): seen[fld.name].append(fld.type)
    fields=[]
    for name,types in seen.items():
        real=[t for t in types if not pa.types.is_null(t)]
        if not real: chosen=pa.string()
        elif any(pa.types.is_floating(t) for t in real): chosen=pa.float64()
        else: chosen=real[0]
        fields.append(pa.field(name,chosen))
    return pa.schema(fields)

# Read every part file in a folder with one shared schema, as one DataFrame.
def read_dir(d):
    files=sorted(glob.glob(d+'/*.parquet'))
    sch=unified_schema(files)
    return pa.concat_tables([pq.read_table(f).cast(sch) for f in files]).to_pandas()
os.chdir('/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning')
# Only accounts that are still active or suspended (not deleted, not helpers/victims) are kept.
KEEP=set(pd.read_parquet('data/accused_accounts_not_deleted.parquet').username_key)
print(f'non-deleted accounts: {len(KEEP):,}',flush=True)
summary={}
# Same steps for posts and for comments.
for kind,src,dst,tcol in (('posts','data/arctic_posts','data/history_posts','created_utc'),
                          ('comments','data/arctic_comments','data/history_comments','created_utc')):
    os.makedirs(dst,exist_ok=True)
    # Empty the output folder first so stale part files from an earlier run cannot survive.
    for f in os.listdir(dst): os.remove(os.path.join(dst,f))
    df=read_dir(src)
    raw=len(df)
    # Defect 2: drop rows of deleted or excluded accounts.
    df=df[df.outer_key.isin(KEEP)]
    filt=len(df)
    # Defect 1: the cursor repair appended re-pulled rows, so drop duplicates on (account, item id).
    df=df.drop_duplicates(['outer_key','id'])
    uniq=len(df)
    df=df.sort_values(['outer_key',tcol],kind='mergesort').reset_index(drop=True)
    # Write in chunks of 2 million rows, sorted by account and time.
    CH=2_000_000
    for i in range(0,len(df),CH):
        df.iloc[i:i+CH].to_parquet(f'{dst}/part-{i//CH:04d}.parquet',engine='pyarrow',compression='zstd',index=False)
    print(f'{kind}: {raw:,} raw -> {filt:,} after dropping deleted/excluded -> {uniq:,} after dedupe '
          f'({raw-uniq:,} removed, {(raw-uniq)/raw:.1%})',flush=True)
    # Per-account counts, first/last item time and number of subreddits, recomputed from the deduplicated rows (defect 3).
    g=df.groupby('outer_key')
    s=pd.DataFrame({f'n_{kind}':g.size(),
                    f'first_{kind}_at':g[tcol].min(), f'last_{kind}_at':g[tcol].max(),
                    f'n_subreddits_{kind}':g['subreddit'].nunique()})
    summary[kind]=s
    del df; gc.collect()
# --- Per-account summary across posts and comments ---
S=summary['posts'].join(summary['comments'],how='outer')
# Accounts with no posts (or no comments) get 0, not NaN.
for c in S.columns:
    if c.startswith('n_'): S[c]=S[c].fillna(0).astype('int64')
S['n_items']=S.n_posts+S.n_comments
S['first_activity_at']=S[['first_posts_at','first_comments_at']].min(axis=1)
S['last_activity_at']=S[['last_posts_at','last_comments_at']].max(axis=1)
S['observed_lifespan_days']=(S.last_activity_at-S.first_activity_at)/86400.0
S=S.reset_index().rename(columns={'outer_key':'username_key'})
# Add the expected counts from Arctic Shift and whether the account hit the 10,000-item cap (history_truncated).
led=pd.read_parquet('data/arctic_pull_done.parquet')[['username_key','expected_posts','expected_comments','history_truncated','cap']]
led['history_truncated']=led.history_truncated.fillna(False).astype(bool)
S=S.merge(led,on='username_key',how='left')
S.to_parquet('data/history_summary.parquet',engine='pyarrow',compression='zstd',index=False)
print(f'\nwrote data/history_summary.parquet: {len(S):,} accounts with history',flush=True)
print(S[['n_posts','n_comments','n_items','observed_lifespan_days']].describe().round(1).to_string(),flush=True)
