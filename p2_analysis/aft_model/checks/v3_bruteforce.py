"""Independent check of the v3 window variables: recompute them naively from the raw pilot posts/comments/replies at
random pre-accusation episodes and compare with exact_v3_to_cutoff.parquet. Written separately from features.py
(definitions from its docstring). Usage: python checks/v3_bruteforce.py [n_accounts]"""
import os, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.dirname(HERE)
S = os.path.join(P2, 'sample_998_v4_excl_output')
DAY, W = 86400.0, 7.0
rng = np.random.default_rng(19)
ex = pd.read_parquet(os.path.join(P2, 'aft_model_output', 'exact_v9_to_cutoff.parquet'))
ex = ex[ex.after == 0]
acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
ids = rng.choice(ex.id.unique(), int(sys.argv[1]) if len(sys.argv) > 1 else 40, replace=False)
P = pd.read_parquet(os.path.join(S, 'sample_posts.parquet'), columns=['id', 'outer_key', 'created_utc', 'title', 'selftext', 'is_self', 'edited'])
C = pd.read_parquet(os.path.join(S, 'sample_comments.parquet'), columns=['id', 'outer_key', 'created_utc', 'subreddit', 'link_id', 'parent_id', 'body', 'edited'])
P = P[P.outer_key.isin(ids)].drop_duplicates(['outer_key', 'id']); C = C[C.outer_key.isin(ids)].drop_duplicates(['outer_key', 'id'])
R = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(P2, 'replies_output', '*_parts', '*.parquet'))])
R = R[R.outer_key.isin(ids) & R.id.notna()].drop_duplicates(['outer_key', 'id'])
R = R[R.author.fillna('').str.lower() != R.outer_key]
PA = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(P2, 'parent_authors_output', 'parts', '*.parquet'))]).drop_duplicates('parent_id').set_index('parent_id').parent_author


def med_gap(ts):
    ts = sorted(ts)
    g = [(b - a) * 24 for a, b in zip(ts, ts[1:])]
    return np.log1p(np.median(g)) if g else np.log1p(168.0)


def kimjo(ts):
    ts = sorted(ts)
    if len(ts) < 3:
        return 0.0
    g = np.diff(ts); n = len(g); mu, sd = g.mean(), g.std()
    if mu <= 0:
        return 0.0
    r = min(sd / mu, np.sqrt(n - 1))
    return (np.sqrt(n + 1) * r - np.sqrt(n - 1)) / ((np.sqrt(n + 1) - 2) * r + np.sqrt(n - 1))


def burst(ts):
    ts = sorted(ts)
    if len(ts) < 3:
        return 0.0
    g = np.diff(ts); mu, sd = g.mean(), g.std()
    return (sd - mu) / (sd + mu) if sd + mu > 0 else 0.0


SH = pd.read_parquet(os.path.join(P2, 'shared_text_output', 'shared_text_flags_v1.parquet'))
SH = SH[SH.shared & SH.outer_key.isin(ids)]
bad = {}
for k in ids:
    a = acc.loc[k]; t0, T = a.first_action_utc, a.first_mention_adj_utc
    p = P[(P.outer_key == k) & (P.created_utc < T)]; c = C[(C.outer_key == k) & (C.created_utc < T)]
    pt, ct = (p.created_utc - t0).to_numpy() / DAY, (c.created_utc - t0).to_numpy() / DAY
    ptxt = (p.title.fillna('') + ' ' + p.selftext.fillna('').where(~p.selftext.isin(['[removed]', '[deleted]']), '')).str.strip().str.len().to_numpy()
    ctxt = c.body.fillna('').where(~c.body.isin(['[removed]', '[deleted]']), '').str.len().to_numpy()
    r = R[(R.outer_key == k) & (R.created_utc < T)]
    mine = set('t3_' + p.id) | set('t1_' + c.id)
    direct = r[r.parent_id.isin(mine)]
    # interactions: (time, partner, subreddit)
    inter = [((x.created_utc - t0) / DAY, str(PA.get(x.parent_id, '')).lower(), x.subreddit) for x in c.itertuples()]
    inter += [((x.created_utc - t0) / DAY, str(x.author).lower(), x.subreddit) for x in direct.itertuples()]
    inter = [i for i in inter if i[1] not in ('automoderator', '[deleted]', 'none', 'nan', '', k)]
    # alternating chains
    rpar = dict(zip(r.id, r.parent_id))
    turns = {}
    for x in c.sort_values('created_utc').itertuples():
        n = 1
        if isinstance(x.parent_id, str) and x.parent_id.startswith('t1_') and x.parent_id[3:] in rpar:
            gp = rpar[x.parent_id[3:]]
            if gp.startswith('t1_') and gp[3:] in turns:
                n = turns[gp[3:]] + 1
            elif gp in mine and gp.startswith('t3_'):
                n = 2
        turns[x.id] = n
    tv = np.array([turns[i] for i in c.id])
    entries = []
    for _, s in c.sort_values('created_utc').groupby('link_id'):
        last = None
        for x in (s.created_utc.to_numpy() - t0) / DAY:
            if last is None or x - last > W:
                entries.append(x)
            last = x
    entries = np.array(entries)
    e = ex[ex.id == k]
    for row in e.sample(min(25, len(e)), random_state=1).itertuples():
        g = row.start
        inw = lambda ts: (ts > g - W) & (ts <= g)
        exp = {'pace_posts': med_gap(pt[inw(pt)]), 'pace_comments': med_gap(ct[inw(ct)]), 'burstiness': burst(pt[inw(pt)]),
               'burstiness_threads': burst(entries[inw(entries)]) if len(entries) else 0.0,
               'burstiness_all': kimjo(np.r_[pt[inw(pt)], entries[inw(entries)] if len(entries) else []]),
               'chars_posts': ptxt[inw(pt)].mean() if inw(pt).any() else 0.0,
               'chars_comments': ctxt[inw(ct)].mean() if inw(ct).any() else 0.0,
               'thread_turns': tv[inw(ct)].max() if inw(ct).any() else 0}
        allt = np.sort(np.r_[pt, ct]); gaps = np.r_[0.0, np.diff(allt)]
        tl = p.title.fillna('').str.len().to_numpy()
        isself = p.is_self.astype(str).eq('True').to_numpy()
        bl = p.selftext.fillna('').where(~p.selftext.isin(['[removed]', '[deleted]']), '').str.len().to_numpy()
        exp['chars_title'] = tl[inw(pt)].mean() if inw(pt).any() else 0.0
        sw = inw(pt) & isself
        exp['chars_body'] = bl[sw].mean() if sw.any() else 0.0
        exp['return_gap'] = gaps[inw(allt)].max() if inw(allt).any() else 0.0
        st = (SH.utc[(SH.outer_key == k) & (SH.utc < T)].to_numpy() - t0) / DAY
        exp['shared_text'] = int(inw(st).sum())
        hop = []
        cw = c[inw(ct)].sort_values('created_utc')
        for _, s in cw.groupby('subreddit'):
            tt, th = (s.created_utc.to_numpy() - t0) / DAY, s.link_id.to_numpy()
            hop += [(tt[i] - tt[i - 1]) * 24 for i in range(1, len(tt)) if th[i] != th[i - 1]]
        exp['hopping'] = np.log1p(np.median(hop)) if hop else np.log1p(168.0)
        cnt = {}
        for ti, who, _ in inter:
            if g - W < ti <= g:
                cnt[who] = cnt.get(who, 0) + 1
        exp['repeat_partners'] = sum(v >= 2 for v in cnt.values())
        subs = {}
        for ti, who, sb in inter:
            if ti <= g:
                subs.setdefault(who, set()).add(sb)
        exp['cross_sub_partners'] = sum(len(v) >= 2 for v in subs.values())
        for var, v in exp.items():
            if not np.isclose(getattr(row, var), v, atol=1e-6):
                bad.setdefault(var, []).append((k, g, getattr(row, var), v))
print('checked', len(ids), 'accounts')
for var in ['pace_posts', 'pace_comments', 'burstiness', 'burstiness_threads', 'chars_posts', 'chars_comments', 'thread_turns', 'hopping', 'repeat_partners', 'cross_sub_partners', 'return_gap', 'chars_title', 'chars_body', 'burstiness_all', 'shared_text']:
    print(f'{var:20s} mismatches {len(bad.get(var, []))}', bad.get(var, [])[:3])
