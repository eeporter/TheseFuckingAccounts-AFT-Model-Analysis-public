"""Fixed follow-up horizon design for right truncation (pilot, 991 accounts, 9 Oct covariate set, Weibull, robust SEs).

Pool rule 5 keeps accounts accused before 2022-11-30, so later cohorts can only enter with short times to accusation
(tau_i = cutoff - first action). With a horizon H, keep accounts whose first action is at least H days before the cutoff
and that were accused within H days. Every kept account then had the same window, so being in the sample means T <= H
for every cohort, and start year is no longer tied to how much follow-up an account had. The model describes time to
accusation among accounts accused within H days of first activity.

Inputs: aft_model_output/exact_v1_to_cutoff.parquet (fit_truncation.prep's default) and
sample_998_v4_excl_output/sample_accounts.parquet. Each pickle holds H, path, summary, loglik, params, cov, n_status
and, for the corrected jobs, F_H (each account's F(Psi(H))). checks/summarise_truncation.py tabulates them.
horizon_sample() is also used by fit_main.py and fit_full.py.

Jobs (python fit_horizon.py <job>; each writes aft_model_output/fit_horizon_<job>.pkl):
  H365_none         H = 1 year, no further correction
  H365_observed     H = 1 year, plus -log F(Psi(H)) with the archived behaviour from T to H (all before the cutoff)
  H365_typical      same, behaviour after T held at the account's average before T
  H365_none_bands   H = 1 year, no correction, with the six age bands
  H730_none         H = 2 years, no correction
  H730_observed     H = 2 years, observed path
"""
import os, sys, pickle, time
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(v, '1')
import numpy as np
import pandas as pd

MY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MY)
import fit_truncation as FT                  # prep, covariates, path, BANDS; also patches tvaft's fallback optimiser
from truncation import TruncatedAFT
from tvaft import TimeVaryingAFT

CUTOFF = 1669766400                          # 2022-11-30 00:00 UTC, the pre-ChatGPT cutoff
JOBS = {'H365_none': (365, 'none', False), 'H365_observed': (365, 'observed', False), 'H365_typical': (365, 'typical', False),
        'H365_none_bands': (365, 'none', True), 'H730_none': (730, 'none', False), 'H730_observed': (730, 'observed', False)}
# job -> (H in days, after-T path for the correction or 'none', age bands)


def horizon_sample(ex, acc, H):
    """Episodes of the accounts in the H-day window.

    ex: episodes (id, start, stop in days since first action, after, ...); acc: sample_accounts indexed by
    username_key (first_action_utc, days_to_accused); H: horizon in days. Keeps accounts with tau >= H and accused
    within H days, drops episodes starting at or after H and cuts the last one at H, so the after-T rows used by the
    truncation correction run to H rather than to the cutoff. Returns a copy.
    """
    tau = (CUTOFF - acc['first_action_utc']) / 86400.0
    keep = acc.index[(tau >= H) & (acc['days_to_accused'] <= H)]
    e = ex[ex.id.isin(keep) & (ex.start < H)].copy()
    e['stop'] = np.minimum(e['stop'], H)
    return e


def run(job):
    """Fit one job (see JOBS) and pickle it to aft_model_output/fit_horizon_<job>.pkl."""
    H, how, bands = JOBS[job]
    out_path = os.path.join(FT.OUT, f'fit_horizon_{job}.pkl')
    assert not os.path.exists(out_path), f'exists, not overwritten: {out_path}'
    t = time.time()
    acc = pd.read_parquet(os.path.join(FT.S, 'sample_accounts.parquet')).set_index('username_key')
    ex = horizon_sample(FT.prep(), acc, H)
    covs = [c for c in FT.covariates(ex, FT.REF, bands) if ex.loc[ex.after == 0, c].std() > 0]   # drop constant columns
    subj = ex[ex.after == 0].groupby('id', sort=False)['event'].last().to_frame()
    base = TimeVaryingAFT('weibull').fit(ex.loc[ex.after == 0, ['id', 'start', 'stop'] + covs], subj, covs, robust=True)
    m = base
    if how != 'none':
        # corrected refit (log f(T) - log F(Psi(H))), started from the plain fit; after-T rows of zero length dropped
        e = FT.path(ex, how)
        e = e[(e.after == 0) | (e.stop > e.start)]
        m = TruncatedAFT('weibull').fit(e[['id', 'start', 'stop', 'after'] + covs], subj, covs, robust=True, init=base.params_)
    out = {'name': job, 'H': H, 'path': how, 'summary': m.summary, 'loglik': m.loglik_, 'conv': m.converged_, 'n': m.n_subjects_,
           'params': m.params_, 'cov': m.cov_, 'cov_model': m.cov_model_, 'secs': time.time() - t,
           'n_status': acc.loc[subj.index, 'account_status'].value_counts().to_dict()}
    if how != 'none':
        out['F_H'] = m.F_tau()
    pickle.dump(out, open(out_path, 'wb'))
    print(job, 'n', m.n_subjects_, 'loglik', round(m.loglik_, 2), 'converged', m.converged_, round(time.time() - t), 's', flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
