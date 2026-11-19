#!/usr/bin/env python3
"""Per-text tally of the full copy search (fastscan2 full run), 2026-11-17. Reads the pair files and the query list;
writes two small tables for the AFT model plus a count-only summary. No comment text is read or written.

Inputs:
  --pairs DIR   full_out_v1/pairs_task*_p*.parquet  (one row per accused text x EARLIER archive item, Jaccard >= 0.7)
  --lsh DIR     candidates_v1.parquet + sample_accounts_dates_v1.parquet (to rebuild the list of searched texts with
                exactly the rules of the run: --query pre2022, --minchar 50)
Outputs (in --out):
  searched_texts_v1.parquet   EVERY searched accused text, matched or not (the denominator):
      ckey, kind, outer_key (the ACCUSED account's key), q_created_utc, q_acc_utc, q_vis_len,
      searched_100, searched_140 (q_vis_len >= 100 / 140), pre_accusation (q_created_utc < q_acc_utc)
  per_text_tally_v1.parquet   one row per searched text that has at least one earlier match at 50 characters:
      ckey, q_created_utc, q_acc_utc, q_vis_len, and for S in (50, 100, 140):
        n_copies_S          earlier archive items matching (at 100 / 140: both visible lengths >= 100 / 140)
        n_authors_S         distinct authors of those items (deleted authors count as one)
        n_self_copies_S     of n_copies_S, items written by the accused account itself
        n_deleted_copies_S  of n_copies_S, items whose author is [deleted]
        n_other_copies_S    n_copies_S minus self minus deleted
        n_other_authors_S   distinct authors excluding the account itself and [deleted]
        n_subreddits_S      distinct subreddits of those items
        first_seen_utc_S    earliest a_created_utc among those items
  tally_summary_v1.json       counts only
Neither output contains comment text or the usernames of matched (archive) authors; searched_texts has the accused
account key, which the AFT data already holds. Keep both off any public place.

Job: run_tally_fast2_v1.sh, job 43331456 (on full_out_v1 of array 43329112 and lsh/).
Self / deleted / other: an archive author is "deleted" if it is [deleted], [removed], empty or null; "self" if it equals
the accused account's key (case-insensitive) and is not deleted; "other" otherwise. The 100 and 140 tallies keep only
pairs where BOTH visible lengths reach the threshold, matching what a search at that minchar would have found
(the run used minchar 50 on both sides)."""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules2 as R

ap = argparse.ArgumentParser()
ap.add_argument('--pairs', required=True); ap.add_argument('--lsh', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--minchar', type=int, default=50); ap.add_argument('--query', default='pre2022')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T = {'minchar': a.minchar, 'query': a.query}

# ---- every searched text (same selection code as the run) ----
# all candidate texts (for kind and outer_key per ckey); Q = the texts actually searched, same rules as the run
c = pq.read_table(f'{a.lsh}/candidates_v1.parquet', columns=['kind', 'id', 'outer_key', 'created_utc', 'text'])
Q = R.Queries(a.lsh, a.query, a.minchar)
allkey = np.array([f'{k}:{i}' for k, i in zip(c['kind'].to_pylist(), c['id'].to_pylist())], dtype=object)
meta = pd.DataFrame({'ckey': allkey, 'kind': c['kind'].to_pylist(), 'outer_key': c['outer_key'].to_pylist()}).drop_duplicates('ckey')
S = pd.DataFrame({'ckey': Q.ckey.astype(str), 'q_created_utc': Q.created.astype(np.int64), 'q_acc_utc': Q.acc.astype(np.int64),
                  'q_vis_len': np.asarray(Q.vis).astype(np.int64)})
T['searched_texts'] = len(S); T['searched_ckey_distinct'] = int(S.ckey.nunique())
S = S.merge(meta, on='ckey', how='left')
assert S.outer_key.notna().all()
S['searched_100'] = S.q_vis_len >= 100
S['searched_140'] = S.q_vis_len >= 140
S['pre_accusation'] = S.q_created_utc < S.q_acc_utc
S = S[['ckey', 'kind', 'outer_key', 'q_created_utc', 'q_acc_utc', 'q_vis_len', 'searched_100', 'searched_140', 'pre_accusation']]
pq.write_table(pa.Table.from_pandas(S, preserve_index=False), f'{a.out}/searched_texts_v1.parquet')
T['searched_accounts'] = int(S.outer_key.nunique())
T['searched_140'] = int(S.searched_140.sum()); T['searched_140_accounts'] = int(S[S.searched_140].outer_key.nunique())
T['searched_100'] = int(S.searched_100.sum()); T['searched_100_accounts'] = int(S[S.searched_100].outer_key.nunique())
T['searched_pre_accusation_100'] = int((S.pre_accusation & S.searched_100).sum())
T['searched_pre_accusation_100_accounts'] = int(S[S.pre_accusation & S.searched_100].outer_key.nunique())
T['searched_pre_accusation_50'] = int(S.pre_accusation.sum()); T['searched_pre_accusation_140'] = int((S.pre_accusation & S.searched_140).sum())
T['searched_pre_accusation_140_accounts'] = int(S[S.pre_accusation & S.searched_140].outer_key.nunique())

# ---- pairs ----
files = sorted(glob.glob(f'{a.pairs}/pairs_task*_p*.parquet'))
T['pair_files'] = len(files)
cols = ['ckey', 'aid', 'akind', 'a_author', 'a_subreddit', 'a_created_utc', 'q_created_utc', 'q_vis_len', 'a_vis_len']
P = pa.concat_tables([pq.read_table(f, columns=cols) for f in files]).to_pandas()
T['pairs'] = len(P)
T['pairs_duplicate_ckey_aid_akind'] = int(P.duplicated(['ckey', 'aid', 'akind']).sum())
T['pairs_not_earlier'] = int((P.a_created_utc >= P.q_created_utc).sum())          # must be 0
T['pairs_ckey_not_searched'] = int((~P.ckey.isin(S.ckey)).sum())                  # must be 0
P = P.merge(S[['ckey', 'outer_key']], on='ckey', how='left')
au = P.a_author.fillna('[deleted]').astype(str)
# classify the archive author of each pair (see module docstring)
P['is_del'] = au.str.lower().isin(['[deleted]', '[removed]', ''])
P['is_self'] = (au.str.lower() == P.outer_key.astype(str).str.lower()) & ~P.is_del
P['is_other'] = ~P.is_del & ~P.is_self
P['a_author'] = au
P['is140'] = (P.q_vis_len >= 140) & (P.a_vis_len >= 140)
P['is100'] = (P.q_vis_len >= 100) & (P.a_vis_len >= 100)


def tally(D, s):
    """Per-ckey counts for pairs table D at length threshold s (column names get suffix _s): copies, distinct authors,
    self / deleted / other copies, distinct other authors, distinct subreddits, earliest archive time."""
    g = D.groupby('ckey', sort=False)
    o = pd.DataFrame({f'n_copies_{s}': g.size(), f'n_authors_{s}': g.a_author.nunique(),
                      f'n_self_copies_{s}': g.is_self.sum(), f'n_deleted_copies_{s}': g.is_del.sum(),
                      f'n_other_copies_{s}': g.is_other.sum(), f'n_subreddits_{s}': g.a_subreddit.nunique(),
                      f'first_seen_utc_{s}': g.a_created_utc.min()})
    o[f'n_other_authors_{s}'] = D[D.is_other].groupby('ckey', sort=False).a_author.nunique().reindex(o.index).fillna(0)
    return o


# inner join at 50 (texts with any match); left joins at 100 / 140, missing counts become 0
t50, t100, t140 = tally(P, 50), tally(P[P.is100], 100), tally(P[P.is140], 140)
Tt = S[['ckey', 'q_created_utc', 'q_acc_utc', 'q_vis_len']].merge(t50, left_on='ckey', right_index=True, how='inner')
Tt = Tt.merge(t100, left_on='ckey', right_index=True, how='left')
Tt = Tt.merge(t140, left_on='ckey', right_index=True, how='left')
for col in Tt.columns:
    if col.startswith('n_'): Tt[col] = Tt[col].fillna(0).astype(np.int64)
    if col.startswith('first_seen'): Tt[col] = Tt[col].astype('Int64')
order = ['ckey', 'q_created_utc', 'q_acc_utc', 'q_vis_len'] + [f'{n}_{s}' for s in (50, 100, 140) for n in
         ('n_copies', 'n_authors', 'n_self_copies', 'n_deleted_copies', 'n_other_copies', 'n_other_authors', 'n_subreddits', 'first_seen_utc')]
Tt = Tt[order]
pq.write_table(pa.Table.from_pandas(Tt, preserve_index=False), f'{a.out}/per_text_tally_v1.parquet')

acct = S.set_index('ckey').outer_key
pre = S.set_index('ckey').pre_accusation
# summary per threshold; the denominator at 100 / 140 is the searched texts with q_vis_len >= 100 / 140
for s in (50, 100, 140):
    m = Tt[Tt[f'n_copies_{s}'] > 0]
    base = S if s == 50 else S[S[f'searched_{s}']]
    d = {'texts_searched': len(base), 'texts_with_earlier_copy': len(m), 'accounts_with_earlier_copy': int(acct.loc[m.ckey].nunique()),
         'pairs': int(m[f'n_copies_{s}'].sum()),
         'texts_with_other_author_copy': int((m[f'n_other_authors_{s}'] > 0).sum()),
         'accounts_with_other_author_copy': int(acct.loc[m[m[f'n_other_authors_{s}'] > 0].ckey].nunique()),
         'texts_with_only_self_or_deleted_copies': int((m[f'n_other_authors_{s}'] == 0).sum()),
         'pre_accusation_texts_searched': int(base.pre_accusation.sum()),
         'pre_accusation_texts_with_other_author_copy': int((pre.loc[m.ckey].values & (m[f'n_other_authors_{s}'] > 0).values).sum()),
         'n_copies_quantiles_50_90_99_max': [int(x) for x in m[f'n_copies_{s}'].quantile([.5, .9, .99, 1]).tolist()] if len(m) else [],
         'n_other_authors_quantiles_50_90_99_max': [int(x) for x in m[f'n_other_authors_{s}'].quantile([.5, .9, .99, 1]).tolist()] if len(m) else [],
         'texts_by_n_other_authors': {k: int(((m[f'n_other_authors_{s}'] >= lo) & (m[f'n_other_authors_{s}'] <= hi)).sum())
                                      for k, lo, hi in (('0', 0, 0), ('1', 1, 1), ('2-4', 2, 4), ('5-19', 5, 19), ('20-99', 20, 99), ('100-999', 100, 999), ('1000+', 1000, 10**12))},
         'share_of_pairs_in_top_10_texts': round(float(m[f'n_copies_{s}'].nlargest(10).sum() / max(1, m[f'n_copies_{s}'].sum())), 4),
         'share_of_pairs_in_top_100_texts': round(float(m[f'n_copies_{s}'].nlargest(100).sum() / max(1, m[f'n_copies_{s}'].sum())), 4)}
    T[f'at_{s}'] = d
T['pairs_by_akind'] = {k: int(v) for k, v in P.akind.value_counts().items()}
json.dump(T, open(f'{a.out}/tally_summary_v1.json', 'w'), indent=1)
print(json.dumps(T, indent=1))
