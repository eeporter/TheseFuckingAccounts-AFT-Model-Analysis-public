"""Right-truncation-corrected time-varying AFT (Lagakos, Barraj & De Gruttola 1988).

The pilot only contains accounts accused before the cutoff (pool rule 5), i.e. T_i <= tau_i = cutoff - first action.
Each account's likelihood is conditioned on that: log L_i = log L_i(untruncated) - log F(Psi_i(tau_i)), where Psi_i runs
over the account's covariate path up to tau_i. The path after T_i is not part of the event model, so it is an assumption
supplied by the caller as extra episodes (episodes with after = 1).

Same audit approach as audit/refits.py (TruncatedAFT, carry-forward path), generalised to a time-varying path after T_i
and with log F computed so that it stays finite when F is close to 0 or 1.
"""
import os, sys
import numpy as np
import pandas as pd
import autograd.numpy as anp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tvaft
from tvaft import TimeVaryingAFT


def log_F(dist, psi, theta):
    """log of the baseline cdf at clock value psi, finite for psi near 0 or very large"""
    if dist == 'weibull':
        w = (anp.log(psi) - theta[0]) / anp.exp(theta[1])
        s = anp.exp(anp.minimum(w, 30.0))           # -log S; F = 1 to double precision above w = 30
        return anp.where(w < -18.0, w, anp.log(-anp.expm1(-anp.maximum(s, 1e-8))))   # log F = w - s/2 + ... for small s
    _, logS = tvaft.baseline_logf_logS(dist, psi, theta)
    return anp.log(-anp.expm1(anp.minimum(logS, -1e-300)))


class TruncatedAFT(TimeVaryingAFT):
    """fit(episodes, subjects, covariates): episodes need an 'after' column; after = 0 rows are the usual episodes
    (0 to T_i), after = 1 rows the assumed path from T_i to tau_i. Every subject needs at least one after row."""

    def _prep(self, episodes, subjects):
        after = episodes['after'].to_numpy() == 1
        super()._prep(episodes[~after], subjects)
        ea = episodes[after].sort_values(['id', 'start'])
        if set(ea['id']) != set(self.ids_):
            raise ValueError('every subject needs an after-T path')
        tau = ea.groupby('id').stop.max().reindex(self.ids_).to_numpy()
        start = ea.groupby('id').start.min().reindex(self.ids_).to_numpy()
        if not np.allclose(start, self.T_):
            raise ValueError('the after-T path must start at T')
        ea = ea.set_index('id').loc[self.ids_].reset_index()          # same subject order as the main episodes
        first = ~ea['id'].duplicated()
        self._Xa = ea[self.covariates].to_numpy(float)
        if not np.isfinite(self._Xa).all():
            raise ValueError('after-T covariates contain missing or infinite values')
        self._dura = (ea['stop'] - ea['start']).to_numpy(float)
        self._fa = np.flatnonzero(first.to_numpy())
        self._la = np.r_[self._fa[1:] - 1, len(ea) - 1]
        self.tau_ = tau

    def _psi_after(self, beta):
        lp = anp.clip(anp.dot(self._Xa, beta), -60.0, 60.0)        # guards optimiser trial steps against overflow
        cs = anp.cumsum(anp.exp(-lp) * self._dura)
        before = anp.concatenate([anp.zeros(1), cs[:-1]])
        return cs[self._la] - before[self._fa]

    def psi_parts(self, params):
        p = len(self.covariates)
        beta = params[:p]
        psi_T = self._psi(anp.exp(-anp.dot(self._X, beta)) * self._dur)
        return psi_T, psi_T + self._psi_after(beta)

    def _ll_subjects(self, params):
        # same as TimeVaryingAFT (all events here, no delayed entry), with the linear predictor clipped so that
        # optimiser trial steps far from the optimum give finite values instead of inf - inf
        p = len(self.covariates)
        if (1 - self._event).any() or self._has_entry.any():
            raise ValueError('TruncatedAFT handles uncensored subjects without delayed entry only')
        beta = params[:p]
        theta = params[p:]
        lp = anp.clip(anp.dot(self._X, beta), -60.0, 60.0)
        psi_T = self._psi(anp.exp(-lp) * self._dur)
        psi_tau = psi_T + self._psi_after(beta)
        if self.dist == 'weibull':                  # tvaft's Weibull log density with w clipped (finite far from the optimum)
            y = anp.log(psi_T); sig = anp.exp(theta[1])
            w = (y - theta[0]) / sig
            logf = w - anp.exp(anp.minimum(w, 50.0)) - theta[1] - y
        else:
            logf, _ = tvaft.baseline_logf_logS(self.dist, psi_T, theta)
        return logf - lp[self._last_idx] - log_F(self.dist, psi_tau, theta)

    def F_tau(self, params=None):
        params = self.params_ if params is None else params
        _, psi_tau = self.psi_parts(params)
        return pd.Series(np.exp(np.asarray(log_F(self.dist, psi_tau, params[len(self.covariates):]))), index=self.ids_)
