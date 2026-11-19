"""Pilot model fits on the decided sample (2026-10-16): all 11 capped accounts dropped (exact_v2_to_cutoff.parquet),
1-year window (rules.in_window: first action >= 1 year before the cutoff, accused within a year) -> 757 accounts.
9 Oct covariate set, Weibull, exact time, robust (sandwich) SEs.

Jobs (python fit_main.py <job>; each writes aft_model_output/fit_main_<job>.pkl, never overwritten):
  main             the model
  main_bands       + six age-band indicators (1/7/30/180/365 days; the 730+ band is empty in a 1-year window)
  chk_typical      + correction for the within-window selection, behaviour after T at the account's pre-T average
  chk_le2020       main on 2015-2020 starters only (the cohorts the 2-year window keeps, for a like-for-like comparison)
  chk_H730         2-year window instead (drops all 2021 starters too)
  chk_early        main on 2015-2018 starters only (reference year 2018)
Calibration (observed vs expected accusations by account age) is stored for main and main_bands.
"""
import os, sys, pickle, time
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(v, '1')
import numpy as np
import pandas as pd

MY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MY)
import rules
import fit_truncation as FT                  # prep, covariates, path, BANDS; also patches tvaft's fallback optimiser
from fit_horizon import horizon_sample
from truncation import TruncatedAFT
from tvaft import TimeVaryingAFT
import tvaft

SOURCE = 'exact_v2_to_cutoff.parquet'
JOBS = {'main': dict(H=365), 'main_bands': dict(H=365, bands=True), 'chk_typical': dict(H=365, path='typical'),
        'chk_le2020': dict(H=365, years=(2015, 2020)), 'chk_H730': dict(H=730), 'chk_early': dict(H=365, years=(2015, 2018), ref=2018)}
CAL_EDGES = [0.25, 1, 2, 7, 14, 30, 90, 180, 365]


def calibration(m, ex, covs):
    """observed vs expected accusations in age intervals (Weibull): E(h) = sum_i Lambda0(Psi_i(min(h, T_i)))"""
    p = len(covs); beta, (mu, ls) = m.params_[:p], m.params_[p:p + 2]
    pre = ex[ex.after == 0]
    rate = np.exp(-pre[covs].to_numpy(float) @ beta)
    T = pre.groupby('id').stop.max()
    rows = []
    for h in CAL_EDGES:
        mm = pre.start.to_numpy() < h
        psi = pd.Series(rate[mm] * (np.minimum(pre.stop.to_numpy()[mm], h) - pre.start.to_numpy()[mm])).groupby(pre.id.to_numpy()[mm]).sum()
        rows.append({'by_day': h, 'observed': int((T <= h).sum()), 'expected': float(np.exp((np.log(psi) - mu) / np.exp(ls)).sum())})
    c = pd.DataFrame(rows).set_index('by_day'); iv = c.diff().fillna(c)
    iv['O/E'] = iv.observed / iv.expected
    iv['chi2'] = (iv.observed - iv.expected) ** 2 / iv.expected
    return iv


def run(job):
    cfg = JOBS[job]
    out_path = os.path.join(FT.OUT, f'fit_main_{job}.pkl')
    assert not os.path.exists(out_path), f'exists, not overwritten: {out_path}'
    t = time.time()
    acc = pd.read_parquet(os.path.join(FT.S, 'sample_accounts.parquet')).set_index('username_key')
    ex = horizon_sample(FT.prep(source=SOURCE), acc, cfg['H'])
    assert set(ex.id) == set(rules.in_window(acc.loc[ex.id.unique()], cfg['H']))
    if 'years' in cfg:
        ex = ex[ex.start_year.between(*cfg['years'])]
    ref = cfg.get('ref', FT.REF)
    covs = [c for c in FT.covariates(ex, ref, cfg.get('bands', False)) if ex.loc[ex.after == 0, c].std() > 0]
    subj = ex[ex.after == 0].groupby('id', sort=False)['event'].last().to_frame()
    m = TimeVaryingAFT('weibull').fit(ex.loc[ex.after == 0, ['id', 'start', 'stop'] + covs], subj, covs, robust=True)
    out = {'name': job, 'cfg': cfg, 'covariates': covs}
    if cfg.get('path'):
        e = FT.path(ex, cfg['path'])
        e = e[(e.after == 0) | (e.stop > e.start)]
        m0 = m
        m = TruncatedAFT('weibull').fit(e[['id', 'start', 'stop', 'after'] + covs], subj, covs, robust=True, init=m0.params_)
        out['F_H'] = m.F_tau()
    out.update({'summary': m.summary, 'loglik': m.loglik_, 'AIC': m.AIC_, 'conv': m.converged_, 'n': m.n_subjects_,
                'params': m.params_, 'cov': m.cov_, 'cov_model': m.cov_model_, 'secs': time.time() - t,
                'n_status': acc.loc[subj.index, 'account_status'].value_counts().to_dict(),
                'n_by_year': ex[ex.after == 0].groupby('id').start_year.first().value_counts().sort_index().to_dict()})
    if job.startswith('main'):
        out['calibration'] = calibration(m, ex, covs)
    pickle.dump(out, open(out_path, 'wb'))
    print(job, 'n', m.n_subjects_, 'loglik', round(m.loglik_, 2), 'converged', m.converged_, round(time.time() - t), 's', flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
