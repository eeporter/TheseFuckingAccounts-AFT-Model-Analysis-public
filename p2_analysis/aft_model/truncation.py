"""Right-truncation-corrected time-varying AFT (Lagakos, Barraj & De Gruttola 1988).

The pilot only contains accounts accused before the cutoff (pool rule 5), i.e. T_i <= tau_i = cutoff - first action.
Each account's likelihood is conditioned on that: log L_i = log L_i(untruncated) - log F(Psi_i(tau_i)), where Psi_i runs
over the account's covariate path up to tau_i. The path after T_i is not part of the event model, so it is an assumption
supplied by the caller as extra episodes (episodes with after = 1).

Same audit approach as audit/refits.py (TruncatedAFT, carry-forward path), generalised to a time-varying path after T_i
and with log F computed so that it stays finite when F is close to 0 or 1.

Library module, reads and writes no files. Times are days since the account's first action. Use (as in fit_main.py,
fit_horizon.py, fit_truncation.py):
    m = TruncatedAFT('weibull').fit(episodes_with_after, subjects, covs, robust=True, init=untruncated_fit.params_)
    m.F_tau()      # per-account baseline probability of being accused by tau_i under the fitted model
"""
import os, sys
import numpy as np
import pandas as pd
import autograd.numpy as anp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # p2_analysis/, for tvaft
import tvaft
from tvaft import TimeVaryingAFT


def log_F(dist, psi, theta):
    """log of the baseline cdf at clock value psi, finite for psi near 0 or very large

    psi: clock value(s) on the time scale (days), > 0. theta = (mu, log_sigma[, q]). Returns log F0(psi) elementwise.
    """
    if dist == 'weibull':
        w = (anp.log(psi) - theta[0]) / anp.exp(theta[1])
        s = anp.exp(anp.minimum(w, 30.0))           # -log S; F = 1 to double precision above w = 30
        # log F = log(1 - exp(-s)) via expm1 (accurate when s is small). anp.where evaluates both branches, so s is
        # floored at 1e-8 to keep the unused branch finite; below w = -18 (s < 1.5e-8) log F is taken as w (error ~ s/2).
        return anp.where(w < -18.0, w, anp.log(-anp.expm1(-anp.maximum(s, 1e-8))))   # log F = w - s/2 + ... for small s
    # other baselines: log F = log(1 - S) = log(-expm1(log S)); capping log S just below 0 keeps F > 0 so the log is finite
    _, logS = tvaft.baseline_logf_logS(dist, psi, theta)
    return anp.log(-anp.expm1(anp.minimum(logS, -1e-300)))


class TruncatedAFT(TimeVaryingAFT):
    """fit(episodes, subjects, covariates): episodes need an 'after' column; after = 0 rows are the usual episodes
    (0 to T_i), after = 1 rows the assumed path from T_i to tau_i. Every subject needs at least one after row.

    Per-account log-likelihood: log f(T_i) - log F(Psi_i(tau_i)), i.e. the event density conditioned on T_i <= tau_i.
    All subjects must have an event and no delayed entry. fit, robust SEs and summary are inherited from TimeVaryingAFT.
    """

    def _prep(self, episodes, subjects):
        """Run TimeVaryingAFT._prep on the after = 0 rows, then cache the after-T path: _Xa (covariates), _dura
        (durations), _fa/_la (first/last after row per subject, same subject order as ids_) and tau_ (end of the
        path, i.e. the truncation time tau_i, per subject). The after rows are only checked to start at T_i; their
        contiguity is not checked."""
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
        """Clock time accumulated on the assumed path from T_i to tau_i, per subject (same running-total trick as
        TimeVaryingAFT._psi)."""
        lp = anp.clip(anp.dot(self._Xa, beta), -60.0, 60.0)        # guards optimiser trial steps against overflow
        cs = anp.cumsum(anp.exp(-lp) * self._dura)
        before = anp.concatenate([anp.zeros(1), cs[:-1]])
        return cs[self._la] - before[self._fa]

    def psi_parts(self, params):
        """(Psi_i(T_i), Psi_i(tau_i)) per subject at params. Note: the before-T linear predictor is not clipped here,
        unlike in _ll_subjects; the two agree whenever |x . beta| <= 60."""
        p = len(self.covariates)
        beta = params[:p]
        psi_T = self._psi(anp.exp(-anp.dot(self._X, beta)) * self._dur)
        return psi_T, psi_T + self._psi_after(beta)

    def _ll_subjects(self, params):
        """Per-subject log f(T_i) - log F(Psi_i(tau_i)) at params = (beta, theta)."""
        # same as TimeVaryingAFT (all events here, no delayed entry), with the linear predictor clipped so that
        # optimiser trial steps far from the optimum give finite values instead of inf - inf
        p = len(self.covariates)
        if (1 - self._event).any() or self._has_entry.any():
            raise ValueError('TruncatedAFT handles uncensored subjects without delayed entry only')
        beta = params[:p]
        theta = params[p:]
        lp = anp.clip(anp.dot(self._X, beta), -60.0, 60.0)
        psi_T = self._psi(anp.exp(-lp) * self._dur)
        psi_tau = psi_T + self._psi_after(beta)          # clock keeps running along the assumed path to tau_i
        if self.dist == 'weibull':                  # tvaft's Weibull log density with w clipped (finite far from the optimum)
            y = anp.log(psi_T); sig = anp.exp(theta[1])
            w = (y - theta[0]) / sig
            logf = w - anp.exp(anp.minimum(w, 50.0)) - theta[1] - y    # theta[1] = log sigma (the Jacobian term)
        else:
            logf, _ = tvaft.baseline_logf_logS(self.dist, psi_T, theta)
        # - lp[last]: Jacobian dPsi/dT at T_i, as in TimeVaryingAFT; - log F: conditioning on T_i <= tau_i
        return logf - lp[self._last_idx] - log_F(self.dist, psi_tau, theta)

    def F_tau(self, params=None):
        """Baseline cdf at Psi_i(tau_i), i.e. P(T_i <= tau_i) given the assumed path, per subject (Series indexed by id).
        params defaults to the fitted params_."""
        params = self.params_ if params is None else params
        _, psi_tau = self.psi_parts(params)
        return pd.Series(np.exp(np.asarray(log_F(self.dist, psi_tau, params[len(self.covariates):]))), index=self.ids_)
