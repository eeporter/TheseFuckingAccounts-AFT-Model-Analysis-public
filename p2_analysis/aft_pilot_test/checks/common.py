"""Shared setup for the test-run checks: the same covariates and episodes as ../fit.py."""
import os
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(v, '1')
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_analysis')
import tvaft
from tvaft import TimeVaryingAFT

HERE = os.path.dirname(os.path.abspath(__file__))
S = '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_analysis/sample_998_v4_excl_output'
GROUPS = ['profile', 'adult', 'crypto', 'politics', 'karma_farming', 'general', 'news', 'sales', 'location', 'entertainment', 'other']
COUNTS = [f'thr_{g}' for g in GROUPS] + ['links_external', 'visual_media', 'crossposts', 'nsfw_outside', 'edits',
          'down_posts', 'down_comments', 'popular_posts', 'popular_comments', 'replies_on_posts', 'tfa_posts_30d']
REF = 2020
_acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
TRUNCATED = sorted(_acc.index[_acc.history_truncated.fillna(False).astype(bool)])   # 10,000-item cap (11 in the pilot)


def load():
    daily = pd.read_parquet(os.path.join(HERE, '..', 'daily.parquet'))
    acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
    daily['raw_' + 'tfa'] = daily['tfa_posts_30d']
    for c in COUNTS:
        daily[c] = np.log1p(daily[c])
    daily['username_digits'] = daily['id'].map(acc['username'].str.contains(r'\d')).astype(float)
    year = pd.to_datetime(daily['id'].map(acc['first_action_utc']), unit='s').dt.year.clip(lower=2015)
    years = sorted(year.unique())
    for y in years:
        if y != REF:
            daily[f'start_{y}'] = (year == y).astype(float)
    covs = COUNTS + ['flooding', 'has_posted', 'has_commented', 'username_digits'] + [f'start_{y}' for y in years if y != REF]
    return daily, acc, covs


def merge(daily, covs, extra=()):
    """merge consecutive days with identical covariates (likelihood unchanged)"""
    cols = list(covs) + list(extra)
    same = (daily[cols].ne(daily[cols].shift()).any(axis=1) | daily['id'].ne(daily['id'].shift())).cumsum()
    return daily.groupby(same).agg(id=('id', 'first'), start=('start', 'first'), stop=('stop', 'last'),
                                   **{c: (c, 'first') for c in cols}).reset_index(drop=True)


def subjects_for(ep):
    return pd.DataFrame({'event': 1}, index=pd.Index(ep['id'].unique()))


def fit(ep, covs, dist='weibull', robust=True, subjects=None, init=None):
    return TimeVaryingAFT(dist).fit(ep, subjects if subjects is not None else subjects_for(ep), covs, robust=robust, init=init)


def cox_snell(m):
    p = len(m.covariates)
    beta, theta = m.params_[:p], m.params_[p:]
    psi = np.asarray(m._psi(np.exp(-m._X @ beta) * m._dur))
    _, logS = tvaft.baseline_logf_logS(m.dist, psi, theta)
    return pd.Series(-np.asarray(logS), index=m.ids_)


def scores(m):
    h = 1e-5
    return np.column_stack([(m._ll_subjects(m.params_ + h * e) - m._ll_subjects(m.params_ - h * e)) / (2 * h)
                            for e in np.eye(len(m.params_))])


# Check-harness only (tvaft.py untouched): when BFGS fails, polish with a trust-region Newton method using the exact
# Hessian instead of tvaft's Nelder-Mead fallback, which takes > 20 min with ~37 parameters.
import types
import autograd
from scipy import optimize as _opt
FALLBACKS = []


def _minimize(fun, x0, jac=None, method=None, options=None, **kw):
    if method == 'BFGS':
        r = _opt.minimize(fun, x0, jac=jac, method='BFGS', options=options)
        if r.success:
            return r
        start = r.x if np.isfinite(r.fun) else x0
        r2 = _opt.minimize(fun, start, jac=jac, hess=autograd.hessian(fun), method='trust-exact', options={'maxiter': 300, 'gtol': 1e-6})
        FALLBACKS.append(bool(r2.success))
        return r2
    return _opt.minimize(fun, x0, jac=jac, method=method, options=options, **kw)


tvaft.optimize = types.SimpleNamespace(minimize=_minimize)
