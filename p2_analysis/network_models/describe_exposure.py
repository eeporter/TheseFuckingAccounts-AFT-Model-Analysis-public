"""Step 2 descriptives and a quick exploratory check (not the AFT) on network_models_output/edges_757.parquet.

Main tie definition = direct replies (direct_reply True: comment_parent, comment_reply, and post_thread comments whose
parent is the account's own post or comment), made before the account's own (edit-adjusted) accusation. Thread
co-presence (all post_thread rows) is reported as a secondary definition.
  partner accused          partner is in accused_accounts_final (74,812), whenever accused
  accused before own       partner_acc_utc < the account's first_mention_adj_utc
  prior exposure           a pre-accusation tie made when the partner had already been accused (partner_acc_utc < reply)
Writes to network_models_output/ (never overwrites) and the figure to p2_figures/network_models_step2/.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Paths: P2 = p2_analysis/; tables go to network_models_output/, the figure to p2_figures/network_models_step2/.
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
OUT = os.path.join(P2, 'network_models_output')
FIG = os.path.join(os.path.dirname(P2), 'p2_figures', 'network_models_step2')
os.makedirs(FIG, exist_ok=True)


# Write a CSV to network_models_output/, refusing to overwrite.
def save_new(df, name, index=False):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        raise FileExistsError(path)
    df.to_csv(path, index=index)


acc = pd.read_parquet(os.path.join(P2, 'sample_998_v4_excl_output', 'sample_accounts.parquet')).set_index('username_key')
e = pd.read_parquet(os.path.join(OUT, 'edges_757.parquet'))
sample = pd.read_parquet(os.path.join(OUT, 'edges_757.parquet'), columns=['sample_account']).sample_account.unique()
# accounts with no kept edge at all still belong to the 757: take the list from the build log's sample
sys.path.insert(0, os.path.join(P2, 'aft_model'))
import rules  # noqa: E402
# The 757 accounts of the decided AFT sample (all capped accounts dropped, 1-year window).
v8 = set(pd.read_parquet(os.path.join(P2, 'aft_model_output', 'exact_v8_to_cutoff.parquet'), columns=['id']).id)
ids = pd.Index(sorted(rules.in_window(acc.loc[sorted(v8)], rules.HORIZON_DAYS)))
assert len(ids) == 757 and set(sample) <= set(ids)
a = acc.loc[ids, ['account_status', 'days_to_accused', 'first_mention_adj_utc']]


# Per-account counts over pre-accusation edges: ties, partners, threads (contacts), accused partners,
# partners accused before the account's own accusation, and partners already accused at the time of the tie (prior exposure).
def per_account(d, tag):
    d = d[d.pre_own_accusation]
    g = d.groupby('sample_account')
    acc_p = d[d.partner_accused]
    r = pd.DataFrame({
        f'{tag}_ties': g.size(),
        f'{tag}_partners': g.partner.nunique(),
        f'{tag}_contacts': g.link_id.nunique(),
        f'{tag}_accused_partners': acc_p.groupby('sample_account').partner.nunique(),
        f'{tag}_partners_accused_before_own': d[d.partner_accused_before_own].groupby('sample_account').partner.nunique(),
        f'{tag}_prior_exposed_partners': d[d.prior_exposure].groupby('sample_account').partner.nunique(),
        f'{tag}_prior_exposed_ties': d[d.prior_exposure].groupby('sample_account').size(),
    }).reindex(ids).fillna(0).astype(int)
    r[f'{tag}_any_prior_exposure'] = r[f'{tag}_prior_exposed_partners'] > 0
    return r


# Four tie definitions: all direct replies, outbound only, inbound only, and thread co-presence (all edges).
direct = e[e.direct_reply]
pa = pd.concat([per_account(direct, 'direct'),
                per_account(direct[direct.direction == 'out'], 'out'),
                per_account(direct[direct.direction == 'in'], 'in'),
                per_account(e, 'thread')], axis=1)
pa = a.join(pa)
pa.index.name = 'sample_account'
save_new(pa.reset_index(), 'per_account_exposure.csv')

# --- distributions by status -------------------------------------------------------------------------------------
cols = [c for c in pa.columns if c.startswith(('direct_', 'out_', 'in_', 'thread_')) and not c.endswith('exposure')]
rows = []
for st, g in [('all', pa)] + list(pa.groupby('account_status')):
    for c in cols:
        x = g[c]
        rows.append({'status': st, 'measure': c, 'n': len(x), 'mean': x.mean(), 'p25': x.quantile(.25),
                     'median': x.median(), 'p75': x.quantile(.75), 'p90': x.quantile(.9), 'max': x.max(),
                     'share_gt0': (x > 0).mean()})
dist = pd.DataFrame(rows)
save_new(dist, 'exposure_distributions_by_status.csv')

# --- timing of partner accusation relative to own (pairs with a pre-accusation direct tie) -------------------------
pairs = (direct[direct.pre_own_accusation & direct.partner_accused]
         .groupby(['sample_account', 'partner'])
         .agg(diff_days=('partner_minus_own_days', 'first'), first_tie_utc=('created_utc', 'min'),
              partner_acc_utc=('partner_acc_utc', 'first'), sample_status=('sample_status', 'first'),
              partner_status=('partner_status', 'first'), ties=('item_id', 'size'), contacts=('link_id', 'nunique'))
         .reset_index())
pairs['accused_before_first_tie'] = pairs.partner_acc_utc < pairs.first_tie_utc
save_new(pairs, 'accused_partner_pairs.csv')
edges_ = [-np.inf, -3650, -1825, -730, -365, -180, -90, -30, 0, 30, 90, 180, 365, 730, 1825, np.inf]
lab = ['<-10y', '-10y..-5y', '-5y..-2y', '-2y..-1y', '-1y..-180d', '-180..-90d', '-90..-30d', '-30..0d', '0..30d',
       '30..90d', '90..180d', '180d..1y', '1y..2y', '2y..5y', '>5y']
pairs['bin'] = pd.cut(pairs.diff_days, edges_, labels=lab, right=False)
hist = pairs.groupby(['bin', 'sample_status']).size().unstack(fill_value=0)
hist['all'] = hist.sum(axis=1)
save_new(hist, 'partner_accusation_timing_hist.csv', index=True)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    v = pairs.diff_days.clip(-3650, 3650)
    ax.hist([v[pairs.sample_status == 'suspended'], v[pairs.sample_status == 'active']], bins=60, stacked=True,
            label=['suspended', 'active'])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('partner accusation − own accusation (days, clipped at ±10 y)')
    ax.set_ylabel('account–partner pairs')
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(FIG, 'partner_accusation_timing.png')
    if not os.path.exists(p):
        fig.savefig(p, dpi=150)
except Exception as ex:  # figure is optional
    print('figure skipped:', ex)

# --- exploratory: days_to_accused by any prior exposure (NOT the AFT) --------------------------------------------
# Accounts split into thirds by number of direct ties, so exposure is compared within similar activity levels.
pa['volume_tertile'] = pd.qcut(pa.direct_ties.rank(method='first'), 3, labels=['low', 'mid', 'high'])
rows = []
for defn in ['direct', 'out', 'in', 'thread']:
    flag = pa[f'{defn}_any_prior_exposure']
    for strat, g in [('all', pa)] + [(f'tertile_{k}', v) for k, v in pa.groupby('volume_tertile')] + \
                    [(f'status_{k}', v) for k, v in pa.groupby('account_status')]:
        f = flag.loc[g.index]
        x, y = g.days_to_accused[f], g.days_to_accused[~f]
        p = mannwhitneyu(x, y, alternative='two-sided').pvalue if len(x) and len(y) else np.nan
        rows.append({'exposure_def': defn, 'stratum': strat, 'n_exposed': len(x), 'n_unexposed': len(y),
                     'median_days_exposed': x.median(), 'median_days_unexposed': y.median(),
                     'mean_ties_exposed': g.direct_ties[f].mean(), 'mean_ties_unexposed': g.direct_ties[~f].mean(),
                     'mannwhitney_p': p})
mw = pd.DataFrame(rows)
save_new(mw, 'exploratory_days_to_accused_by_exposure.csv')
tert = pa.groupby('volume_tertile').direct_ties.agg(['min', 'max', 'size'])

pd.set_option('display.width', 200)
print('censoring check: all 757 accused within the window ->', bool((pa.days_to_accused <= 365).all()))
print('\nvolume tertiles (pre-accusation direct ties):\n', tert)
print('\nany exposure, by definition and status:\n',
      pa.groupby('account_status')[[c for c in pa.columns if c.endswith('any_prior_exposure')]].agg(['sum', 'mean']).T)
print('\naccused-before-own any (direct):', (pa.direct_partners_accused_before_own > 0).sum(),
      ' accused partner ever (direct):', (pa.direct_accused_partners > 0).sum())
print('\nkey distributions (direct):\n', dist[dist.measure.str.startswith('direct_')].round(3).to_string(index=False))
print('\npairs:', len(pairs), ' accused before first tie:', int(pairs.accused_before_first_tie.sum()),
      ' partner accused before own:', int((pairs.diff_days < 0).sum()))
print('\ntiming hist:\n', hist)
print('\npartner status among accused partners:\n', pairs.drop_duplicates('partner').partner_status.value_counts(dropna=False))
print('\nexploratory:\n', mw.round(4).to_string(index=False))
