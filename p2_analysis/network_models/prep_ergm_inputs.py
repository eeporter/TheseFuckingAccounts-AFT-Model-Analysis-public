"""Step 3 inputs: alter-alter networks (ego removed) of the four case studies, from their saved outputs only (nothing is
re-collected). Writes node and edge tables per ego to network_models_output/step3_inputs/ (never overwrites).

Per ego:
  CaseB        alters_min2 (MIN_CONTACTS = 2, as the notebook used) minus text-rule bots; ties = alter_ties
                     (replies between alters inside the ego's threads, weight = distinct shared posts). activity =
                     log(1 + contacts with the ego).
  CaseA ego_alter_ties_output: alter-alter ties from each alter's FULL history (the notebook's design,
                     no MIN_CONTACTS; the min-2 set in ego_net_CaseA_output is empty). activity =
                     log(1 + archived comments + posts) from alter_meta, since ties could form anywhere in the history.
  CaseC        all alters (Full_networks notebook: no MIN_CONTACTS) and their 29 in-thread ties; at MIN_CONTACTS
                     = 2 there are 12 alters and no tie. activity = log(1 + contacts).
  CaseD       all 93 alters, 3 in-thread ties: described, not modelled.
Status = account_status (Active / Suspended / Deleted; 'not checked' if missing).
"""
import os, glob
import numpy as np
import pandas as pd

# Paths: P2 = p2_analysis/; node and tie tables go to network_models_output/step3_inputs/.
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
OUT = os.path.join(P2, 'network_models_output', 'step3_inputs')
os.makedirs(OUT, exist_ok=True)


# Write a CSV, refusing to overwrite.
def save_new(df, name):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        raise FileExistsError(path)
    df.to_csv(path, index=False)


# Save one ego's node table (alter, status, activity, contacts) and tie table (alter-alter pairs, both ends kept as nodes,
# no self-ties), and print size, density and status mix.
def finish(ego, nodes, ties, note):
    nodes = nodes.drop_duplicates('alter').copy()
    nodes['status'] = nodes['status'].fillna('not checked')
    keep = set(nodes.alter)
    ties = ties[ties.a.isin(keep) & ties.b.isin(keep) & (ties.a != ties.b)][['a', 'b', 'posts']]
    save_new(nodes[['alter', 'status', 'activity', 'contacts']], f'nodes_{ego}.csv')
    save_new(ties, f'ties_{ego}.csv')
    n, m = len(nodes), len(ties)
    print(f'{ego}: nodes {n}, ties {m}, density {2 * m / max(n * (n - 1), 1):.5f}, ties posts>=2 {(ties.posts >= 2).sum()}, '
          f'tied nodes {len(set(ties.a) | set(ties.b))}, status {nodes.status.value_counts().to_dict()} | {note}')


# CaseB
d = os.path.join(P2, 'ego_net_CaseB_output')
m2 = pd.read_parquet(os.path.join(d, 'alters_min2_CaseB.parquet'))
allx = pd.read_parquet(os.path.join(d, 'alters_all_CaseB.parquet'), columns=['alter', 'bot_text'])
# Alters flagged by the reply-text bot rule are removed.
textbots = set(allx.alter[allx.bot_text.fillna(False)])
st = pd.read_parquet(os.path.join(d, 'account_status.parquet')).set_index('account').account_status
nodes = m2[~m2.alter.isin(textbots)].assign(status=lambda x: x.alter.map(st), activity=lambda x: np.log1p(x.contacts))
print('tellman text-rule bots in min2:', int(m2.alter.isin(textbots).sum()))
finish('CaseB', nodes, pd.read_parquet(os.path.join(d, 'alter_ties_CaseB.parquet')), 'in-thread ties, MIN_CONTACTS=2')

# CaseA (full histories)
d = os.path.join(P2, 'ego_alter_ties_output')
al = pd.read_parquet(os.path.join(d, 'alters_CaseA.parquet'))
meta = pd.read_parquet(os.path.join(d, 'alter_meta.parquet')).set_index('alter')
nodes = al.assign(status=al.account_status, contacts=al.ego_shared_posts,
                  # Activity = log(1 + the alter's total posts and comments in the archive).
                  activity=np.log1p(al.alter.map(meta.num_comments).fillna(0) + al.alter.map(meta.num_posts).fillna(0)))
print('CaseA alters without full history:', int((~al.history_pulled.astype(bool)).sum()))
e = pd.read_parquet(os.path.join(d, 'edges_CaseA.parquet'))
e = e[e.tie_type == 'alter']
# Undirected ties: each pair stored once with the smaller name first.
ties = pd.DataFrame({'a': np.minimum(e.source, e.target), 'b': np.maximum(e.source, e.target), 'posts': e.weight})
finish('CaseA', nodes, ties, 'full-history ties, all alters')

# CaseC and CaseD (all alters, in-thread ties)
for ego in ['CaseC', 'CaseD']:
    d = os.path.join(P2, f'ego_net_{ego}_output')
    al = pd.read_parquet(os.path.join(d, f'alters_all_{ego}.parquet'))
    al = al[~al.bot_text.fillna(False)]
    nodes = al.assign(status=al.account_status, activity=np.log1p(al.contacts))
    finish(ego, nodes, pd.read_parquet(os.path.join(d, f'alter_ties_all_{ego}.parquet')), 'in-thread ties, all alters')
