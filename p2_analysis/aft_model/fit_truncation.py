"""Right-truncation fits (result 2026-10-15: observed and quiet paths diverge, see README) on the 991-account pilot (exact_v1_to_cutoff.parquet), 9 Oct covariate set (M2: log counts,
'has any' indicators, flooding, has posted/commented, username digits, start year), Weibull, robust SEs.

  none           no correction (same as fit_v1 v1_rules)
  observed       corrected; path from T to tau = the account's archived behaviour (pre-cutoff, after the accusation)
  typical        corrected; behaviour after T held at the account's time-weighted average before T
  quiet          corrected; no behaviour after T (counts, any-indicators and flooding 0)
  observed_bands corrected, observed path, plus the six age-band indicators
  early_none     no correction, first action 2015-2018 only (tau >= 4 years, weak truncation), reference year 2018
  early_observed corrected, observed path, 2015-2018 starters
  observed_no_has corrected, observed path, without has posted/commented (diagnostic)
In every corrected path has posted/commented keep their value at T, and the TFA control, start year, username digits
and age bands follow the calendar / account as observed. Writes aft_model_output/fit_truncation_v2.pkl and a comparison
table fit_truncation_v2_compare.csv (v1 was the first run, where the corrected fits other than 'typical' hit NaNs).

Run: python fit_truncation.py (no job argument; fits all jobs above in parallel, the two uncorrected ones first so the
corrected ones can start from them). Inputs: aft_model_output/exact_v1_to_cutoff.parquet and
sample_998_v4_excl_output/sample_accounts.parquet. The pickle maps job -> fit (or {'name', 'error'} if it failed);
corrected fits carry F_tau, each account's F(Psi(tau)). The csv has time ratios and SEs by job.
Importing this module also loads aft_pilot_test/checks/common.py, which swaps tvaft's fallback optimiser; prep,
covariates, path and the constants are reused by fit_horizon.py, fit_main.py and fit_full.py.
"""
import os, sys, pickle, time
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(v, '1')
from multiprocessing import Pool
import numpy as np
import pandas as pd

MY = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(MY)
sys.path.insert(0, MY); sys.path.insert(0, P2)
from truncation import TruncatedAFT
sys.path.insert(0, os.path.join(P2, 'aft_pilot_test', 'checks'))
import common  # noqa: F401  replaces tvaft's Nelder-Mead fallback with trust-region Newton, as in the 9 Oct checks
from tvaft import TimeVaryingAFT

S = os.path.join(P2, 'sample_998_v4_excl_output')
OUT = os.path.join(P2, 'aft_model_output')
GROUPS = ['profile', 'adult', 'crypto', 'politics', 'karma_farming', 'general', 'news', 'sales', 'location', 'entertainment', 'other']
COUNTS = [f'thr_{g}' for g in GROUPS] + ['links_external', 'visual_media', 'crossposts', 'nsfw_outside', 'edits',
          'down_posts', 'down_comments', 'popular_posts', 'popular_comments', 'replies_on_posts', 'tfa_posts_30d']
ANY = ['visual_media', 'thr_entertainment', 'thr_crypto', 'replies_on_posts']   # the four counts with a 'has any' term
# behaviour that a path after T replaces (not the TFA control, which follows the calendar)
BEHAV = [c for c in COUNTS if c != 'tfa_posts_30d'] + ['flooding'] + [f'any_{c}' for c in ANY]
EDGES = [1, 7, 30, 180, 365, 730]                # age-band edges in days; under 1 day is the reference
BANDS = [f'age_{a}_{b}' for a, b in zip(EDGES, EDGES[1:] + ['inf'])]
REF = 2020                                       # reference start year


def prep(ref=REF, source='exact_v1_to_cutoff.parquet'):
    """Read the episodes and build the 9 Oct covariates.

    source: episode parquet in aft_model_output; ref: not used here (the reference year is chosen in covariates()).
    Adds 'has any' for ANY (from the raw counts), puts COUNTS on log(1 + x), username digits, start year (2014 and
    earlier counted as 2015) with one indicator per year, the age bands by episode start, and fixes has posted /
    commented after T at their value at T. Returns all episodes, pre- and after-accusation.
    """
    ex = pd.read_parquet(os.path.join(OUT, source))
    acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
    for c in ANY:
        ex[f'any_{c}'] = (ex[c] > 0).astype(float)
    for c in COUNTS:
        ex[c] = np.log1p(ex[c])
    ex['username_digits'] = ex['id'].map(acc['username'].str.contains(r'\d')).astype(float)
    year = pd.to_datetime(ex['id'].map(acc['first_action_utc']), unit='s').dt.year.clip(lower=2015)
    ex['start_year'] = year
    for y in sorted(year.unique()):
        ex[f'start_{y}'] = (year == y).astype(float)
    for (a, b), nm in zip(zip(EDGES, EDGES[1:] + [np.inf]), BANDS):
        ex[nm] = ((ex.start >= a) & (ex.start < b)).astype(float)
    pre = ex.after == 0
    # has posted/commented: value at T for the whole after-T path
    last = ex[pre].groupby('id')[['has_posted', 'has_commented']].last()
    for c in ['has_posted', 'has_commented']:
        ex.loc[~pre, c] = ex.loc[~pre, 'id'].map(last[c]).to_numpy()
    return ex


def covariates(ex, ref, bands=False):
    """9 Oct covariate names: log counts, flooding, has posted/commented, username digits, start years except ref
    (one per year present in ex), the four 'has any' terms, and the age bands if bands. Constant columns are dropped
    by the callers."""
    years = sorted(ex.start_year.unique())
    return (COUNTS + ['flooding', 'has_posted', 'has_commented', 'username_digits'] + [f'start_{y}' for y in years if y != ref]
            + [f'any_{c}' for c in ANY] + (BANDS if bands else []))


def path(ex, how):
    """Covariates on the after-T rows (after == 1) for the truncation correction.

    how: 'observed' keeps the archived behaviour after the accusation; 'quiet' sets BEHAV to 0; 'typical' sets each
    BEHAV column to the account's duration-weighted mean over its pre-T episodes (so an any-indicator becomes the share
    of pre-T time with any). Other columns (TFA control, years, digits, bands, has posted/commented) are not touched.
    Returns ex itself for 'observed', otherwise a copy.
    """
    if how == 'observed':
        return ex
    ex = ex.copy()
    after = ex.after == 1
    if how == 'quiet':
        ex.loc[after, BEHAV] = 0.0
    elif how == 'typical':
        pre = ex[~after]
        w = pre.stop - pre.start
        avg = (pre[BEHAV].mul(w, axis=0)).groupby(pre.id).sum().div(w.groupby(pre.id).sum(), axis=0)
        ex.loc[after, BEHAV] = avg.reindex(ex.loc[after, 'id']).to_numpy()
    return ex


# job -> (after-T path, age bands, truncation-corrected)
JOBS = {'none': ('none', False, False), 'observed': ('observed', False, True), 'typical': ('typical', False, True),
        'quiet': ('quiet', False, True), 'observed_bands': ('observed', True, True), 'early_none': ('none', False, False),
        'early_observed': ('observed', False, True), 'observed_no_has': ('observed', False, True)}


def one(name, init=None):
    """Fit one job.

    name: key of JOBS; init: the uncorrected fit's result dict, used as start values for a corrected fit.
    Returns a dict (name, summary, loglik, conv, n, params, cov, cov_model, secs, and F_tau if corrected), or
    {'name', 'error'} if the fit raised, so one failure does not stop the pool.
    """
    how, bands, corrected = JOBS[name]
    t = time.time()
    try:
        ex = prep()
        ref = REF
        if name.startswith('early'):
            ex = ex[ex.start_year <= 2018]; ref = 2018
        covs = [c for c in covariates(ex, ref, bands) if ex.loc[ex.after == 0, c].std() > 0]   # drop constant columns
        if name.endswith('no_has'):                   # diagnostic: without the has posted/commented indicators
            covs = [c for c in covs if c not in ('has_posted', 'has_commented')]
        subj = ex[ex.after == 0].groupby('id', sort=False)['event'].last().to_frame()
        if corrected:
            e = path(ex, how)
            x0 = None
            if init is not None:                      # start from the uncorrected fit; new covariates (bands) at 0
                x0 = np.r_[[init['summary']['coef'].get(c, 0.0) for c in covs], init['params'][-2:]]
            m = TruncatedAFT('weibull').fit(e[['id', 'start', 'stop', 'after'] + covs], subj, covs, robust=True, init=x0)
        else:
            e = ex[ex.after == 0]
            m = TimeVaryingAFT('weibull').fit(e[['id', 'start', 'stop'] + covs], subj, covs, robust=True)
        out = {'name': name, 'summary': m.summary, 'loglik': m.loglik_, 'conv': m.converged_, 'n': m.n_subjects_,
               'params': m.params_, 'cov': m.cov_, 'cov_model': m.cov_model_, 'secs': time.time() - t}
        if corrected:
            out['F_tau'] = m.F_tau()
        print(name, round(m.loglik_, 2), m.converged_, 'n', m.n_subjects_, round(time.time() - t), 's', flush=True)
        return out
    except Exception as err:
        print(name, 'failed', repr(err), flush=True)
        return {'name': name, 'error': repr(err)}


if __name__ == '__main__':
    pkl = os.path.join(OUT, 'fit_truncation_v2.pkl')
    assert not os.path.exists(pkl), f'exists, not overwritten: {pkl}'
    # stage 1: the uncorrected fits; stage 2: every other job, started from 'none' (or 'early_none' for early_*)
    plain = ['none', 'early_none']
    with Pool(len(plain)) as pool:
        res = {r['name']: r for r in pool.map(one, plain, chunksize=1)}
    rest = [k for k in JOBS if k not in plain]
    with Pool(len(rest)) as pool:
        res.update({r['name']: r for r in pool.starmap(one, [(k, res['early_none' if k.startswith('early') else 'none']) for k in rest], chunksize=1)})
    pickle.dump(res, open(pkl, 'wb'))
    ok = {k: r for k, r in res.items() if 'summary' in r}
    tr = pd.DataFrame({k: r['summary']['time_ratio'] for k, r in ok.items()})
    se = pd.DataFrame({k: r['summary']['se'] for k, r in ok.items()}).add_prefix('se_')
    tab = pd.concat([tr, se], axis=1)
    tab.to_csv(os.path.join(OUT, 'fit_truncation_v2_compare.csv'))
    print(tr.round(2).to_string())
    # how strongly each corrected fit says the sample is truncated: F(tau) well below 1 means a large correction
    for k, r in ok.items():
        if 'F_tau' in r:
            print(k, 'accounts with F(tau) < 0.9:', int((r['F_tau'] < 0.9).sum()), 'median F(tau)', round(r['F_tau'].median(), 3))
