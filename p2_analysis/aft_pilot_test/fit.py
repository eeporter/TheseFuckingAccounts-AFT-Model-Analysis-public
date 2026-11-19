import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_analysis')
from tvaft import TimeVaryingAFT
pd.set_option('display.width', 200)

HERE = os.path.dirname(os.path.abspath(__file__))
S = '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_analysis/sample_998_v4_excl_output'
daily = pd.read_parquet(os.path.join(HERE, 'daily.parquet'))
acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')

GROUPS = ['profile', 'adult', 'crypto', 'politics', 'karma_farming', 'general', 'news', 'sales', 'location', 'entertainment', 'other']
COUNTS = [f'thr_{g}' for g in GROUPS] + ['links_external', 'visual_media', 'crossposts', 'nsfw_outside', 'edits',
          'down_posts', 'down_comments', 'popular_posts', 'popular_comments', 'replies_on_posts', 'tfa_posts_30d']
for c in COUNTS:
    daily[c] = np.log1p(daily[c])
daily['username_digits'] = daily['id'].map(acc['username'].str.contains(r'\d')).astype(float)
year = pd.to_datetime(daily['id'].map(acc['first_action_utc']), unit='s').dt.year.clip(lower=2015)
YEARS = sorted(year.unique())
REF = 2020
for y in YEARS:
    if y != REF:
        daily[f'start_{y}'] = (year == y).astype(float)
COVS = COUNTS + ['flooding', 'has_posted', 'has_commented', 'username_digits'] + [f'start_{y}' for y in YEARS if y != REF]

# merge consecutive days with identical covariates
same = (daily[COVS].ne(daily[COVS].shift()).any(axis=1) | daily['id'].ne(daily['id'].shift())).cumsum()
ep = daily.groupby(same).agg(id=('id', 'first'), start=('start', 'first'), stop=('stop', 'last'),
                             **{c: (c, 'first') for c in COVS}).reset_index(drop=True)
subjects = pd.DataFrame({'event': 1}, index=acc.index)
print(f'episodes {len(ep)} (from {len(daily)} days), subjects {len(subjects)}')

# start-year counts
print(year.groupby(daily['id']).first().value_counts().sort_index().to_dict())

# VIF on duration-weighted data
w = (ep.stop - ep.start).to_numpy()
X = ep[COVS].to_numpy(float)
m = (w[:, None] * X).sum(0) / w.sum()
Xc = X - m
cov = (w[:, None] * Xc).T @ Xc / w.sum()
sd = np.sqrt(np.diag(cov))
R = cov / np.outer(sd, sd)
vif = pd.Series(np.diag(np.linalg.inv(R)), index=COVS).round(2)
print('\nVIF (duration-weighted):'); print(vif.sort_values(ascending=False).to_string())
pd.Series(vif).to_csv(os.path.join(HERE, 'vif.csv'))

# share of account-time with each count > 0
print('\nshare of account-days with value > 0 / accounts ever > 0:')
ever = ep.groupby('id')[COVS].max().gt(0).sum()
print(pd.DataFrame({'time_share': (ep[COVS].gt(0).mul(w, axis=0).sum() / w.sum()).round(3), 'accounts_ever': ever}).to_string())

res = {}
for dist in ['weibull', 'lognormal', 'loglogistic', 'gengamma']:
    t = time.time()
    m_ = TimeVaryingAFT(dist).fit(ep, subjects, COVS, robust=True)
    res[dist] = m_
    print(f'\n{dist}: loglik {m_.loglik_:.1f} AIC {m_.AIC_:.1f} converged {m_.converged_} ({time.time() - t:.0f}s)', flush=True)
best = min(res, key=lambda d: res[d].AIC_)
print('\nAIC:', {d: round(r.AIC_, 1) for d, r in res.items()}, 'best', best)
for d, r in res.items():
    r.summary.to_csv(os.path.join(HERE, f'summary_{d}.csv'))
print(res[best].summary.round(3).to_string())
