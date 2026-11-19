"""Accelerated failure time (AFT) regression with time-varying covariates.

Model (Cox & Oakes 1984, ch. 6; Robins & Tsiatis 1992): each subject's covariates are piecewise constant over episodes
[start, stop). The subject's baseline clock is

    Psi(t) = sum over episodes of exp(-x_j . beta) * (time spent in episode j before t)

and S(t | x(.)) = S0(Psi(t)). While a subject has covariate value x its clock runs at exp(-x . beta) times baseline speed,
so exp(beta_k) is a time ratio: > 1 slows the event down, < 1 speeds it up. With constant covariates this is the ordinary
AFT model log T = x . beta + log U, U ~ S0. TV_AFT_validation.ipynb checks it against lifelines and R eha::aftreg
(exact with constant covariates). With time-varying covariates eha 2.12.0's event-density term uses the current-covariate
scaled time instead of the accumulated clock, so its density does not integrate to 1; with that one term patched, eha
matches this implementation exactly.

Log-likelihood per subject: event * [log f0(Psi(T)) - x_last . beta] + (1 - event) * log S0(Psi(T)), minus
log S0(Psi(entry)) for a left-truncated subject (delayed entry).

Baseline S0 for log U = mu + sigma * W:
    weibull      W standard minimum extreme value   (lifelines WeibullAFTFitter: rho = 1 / sigma, lambda = exp(mu))
    lognormal    W standard normal
    loglogistic  W standard logistic
    gengamma     Prentice (1974) generalized gamma with shape q; q = 0 is log-normal, q = 1 is Weibull
"""
import autograd.numpy as anp
from autograd import grad, hessian, make_jvp
from autograd.scipy.special import gammaln
from autograd.scipy.stats import norm
from autograd_gamma import gammainc, gammaincc
import numpy as np
import pandas as pd
from scipy import optimize, stats

DISTRIBUTIONS = ('weibull', 'lognormal', 'loglogistic', 'gengamma')


def baseline_logf_logS(dist, u, theta):
    """Log density and log survival of the baseline clock value u > 0; theta = (mu, log_sigma[, q])."""
    y = anp.log(u)
    mu, sig = theta[0], anp.exp(theta[1])
    w = (y - mu) / sig
    if dist == 'weibull':
        logS = -anp.exp(w)
        logf = w - anp.exp(w) - anp.log(sig) - y
    elif dist == 'lognormal':
        logS = norm.logcdf(-w)
        logf = norm.logpdf(w) - anp.log(sig) - y
    elif dist == 'loglogistic':
        logS = -anp.logaddexp(0.0, w)
        logf = w - 2 * anp.logaddexp(0.0, w) - anp.log(sig) - y
    elif dist == 'gengamma':
        q = theta[2]
        k = q ** -2
        z = k * anp.exp(q * w)
        logf = anp.log(anp.abs(q)) - anp.log(sig) - y - gammaln(k) + k * anp.log(k) + k * (q * w - anp.exp(q * w))
        logS = anp.where(q > 0, anp.log(gammaincc(k, z) + 1e-300), anp.log(gammainc(k, z) + 1e-300))
    else:
        raise ValueError(f'dist must be one of {DISTRIBUTIONS}')
    return logf, logS


def split_at(episodes, times):
    """Split each subject's episode that straddles times[id] (e.g. the entry time) into two, so Psi at that time is exact."""
    t = episodes['id'].map(times)
    straddle = (episodes['start'] < t) & (episodes['stop'] > t)
    before = episodes[straddle].assign(stop=t[straddle])
    after = episodes[straddle].assign(start=t[straddle])
    return (pd.concat([episodes[~straddle], before, after])
              .sort_values(['id', 'start']).reset_index(drop=True))


class TimeVaryingAFT:
    """Parametric AFT regression with piecewise-constant time-varying covariates.

    fit(episodes, subjects, covariates):
        episodes  DataFrame with columns id, start, stop and the covariates; for each subject the episodes must be
                  contiguous and run from 0 to the subject's event/censoring time (one row per subject if covariates
                  never change).
        subjects  DataFrame indexed by id with 'event' (1 = event at the last stop, 0 = right-censored) and optional
                  'entry' (delayed-entry / left-truncation time; an episode boundary must fall on it, see split_at).
        robust    True for sandwich (Huber-White) standard errors, recommended for inference.
    """

    def __init__(self, dist='weibull'):
        if dist not in DISTRIBUTIONS:
            raise ValueError(f'dist must be one of {DISTRIBUTIONS}')
        self.dist = dist

    def _prep(self, episodes, subjects):
        ep = episodes.sort_values(['id', 'start']).reset_index(drop=True)
        if set(ep['id']) != set(subjects.index):
            raise ValueError('episodes and subjects must cover the same ids')
        if not (ep['stop'] > ep['start']).all():
            raise ValueError('every episode needs stop > start')
        nxt = ep.groupby('id')['start'].shift(-1)
        if not np.allclose(nxt.dropna(), ep.loc[nxt.notna(), 'stop']):
            raise ValueError('episodes must be contiguous within a subject')
        first = ~ep['id'].duplicated()
        if not np.allclose(ep.loc[first, 'start'], 0):
            raise ValueError('each subject must start at time 0 (use subjects.entry for delayed entry)')
        order = pd.Index(ep.loc[first, 'id'])
        sub = subjects.loc[order]
        self._X = ep[self.covariates].to_numpy(float)
        if not np.isfinite(self._X).all():
            raise ValueError('covariates contain missing or infinite values')
        self._dur = (ep['stop'] - ep['start']).to_numpy(float)
        self._first_idx = np.flatnonzero(first.to_numpy())
        self._last_idx = np.r_[self._first_idx[1:] - 1, len(ep) - 1]
        self._event = sub['event'].to_numpy(float)
        entry = sub['entry'].to_numpy(float) if 'entry' in sub else np.zeros(len(sub))
        ep_entry = np.repeat(entry, self._last_idx - self._first_idx + 1)
        if 'entry' in sub:
            stops_at_entry = pd.Series(np.isclose(ep['stop'].to_numpy(), ep_entry)).groupby(ep['id'].to_numpy()).any()
            if not stops_at_entry.loc[order[entry > 0]].all():
                raise ValueError('an episode must end exactly at each entry time; use split_at(episodes, entry)')
        self._pre_entry = (ep['stop'].to_numpy(float) <= ep_entry + 1e-9).astype(float)
        self._has_entry = entry > 0
        self.ids_ = order
        self.T_ = ep['stop'].to_numpy()[self._last_idx]

    def _psi(self, contrib):
        cs = anp.cumsum(contrib)
        before = anp.concatenate([anp.zeros(1), cs[:-1]])
        return cs[self._last_idx] - before[self._first_idx]

    def _ll_subjects(self, params):
        p = len(self.covariates)
        beta, theta = params[:p], params[p:]
        contrib = anp.exp(-anp.dot(self._X, beta)) * self._dur
        logf, logS = baseline_logf_logS(self.dist, self._psi(contrib), theta)
        ll = self._event * (logf - anp.dot(self._X[self._last_idx], beta)) + (1 - self._event) * logS
        if self._has_entry.any():
            psi_e = self._psi(contrib * self._pre_entry)
            _, logS_e = baseline_logf_logS(self.dist, anp.where(self._has_entry, psi_e, 1.0), theta)
            ll = ll - anp.where(self._has_entry, logS_e, 0.0)
        return ll

    def _negll(self, params):
        return -anp.sum(self._ll_subjects(params))

    def fit(self, episodes, subjects, covariates, robust=False, init=None):
        self.covariates = list(covariates)
        self._prep(episodes, subjects)
        n_theta = 3 if self.dist == 'gengamma' else 2
        logT = np.log(self.T_)
        x0 = (np.r_[np.zeros(len(self.covariates)), logT.mean(), np.log(logT.std()), [0.5] * (n_theta == 3)]
              if init is None else np.asarray(init, float))
        res = optimize.minimize(self._negll, x0, jac=grad(self._negll), method='BFGS',
                                options={'maxiter': 5000, 'gtol': 1e-6})
        if not res.success:          # BFGS often stops on precision loss at the optimum; polish without gradients
            res = optimize.minimize(self._negll, res.x, method='Nelder-Mead',
                                    options={'maxiter': 40000, 'xatol': 1e-8, 'fatol': 1e-8})
        self.converged_ = bool(res.success)
        self.params_, self.loglik_ = res.x, -res.fun
        H = hessian(self._negll)(res.x)
        H_inv = np.linalg.inv(H)
        self.cov_model_ = H_inv                                   # model-based covariance, kept for comparison
        if robust:                   # per-subject scores, one forward-mode pass per parameter (n_subjects x n_params)
            try:
                jvp = make_jvp(self._ll_subjects)(res.x)
                scores = np.column_stack([jvp(e)[1] for e in np.eye(len(res.x))])
            except NotImplementedError:  # no forward mode for norm.logcdf / autograd_gamma: central differences
                h = 1e-5
                scores = np.column_stack([(self._ll_subjects(res.x + h * e) - self._ll_subjects(res.x - h * e)) / (2 * h)
                                          for e in np.eye(len(res.x))])
            self.cov_ = H_inv @ (scores.T @ scores) @ H_inv
        else:
            self.cov_ = H_inv
        self.robust = robust
        se = np.sqrt(np.diag(self.cov_))
        names = self.covariates + ['mu', 'log_sigma', 'q'][:n_theta]
        self.summary = pd.DataFrame({'coef': res.x, 'se': se, 'time_ratio': np.exp(res.x),
                                     'lower95': np.exp(res.x - 1.96 * se), 'upper95': np.exp(res.x + 1.96 * se),
                                     'p': 2 * stats.norm.sf(np.abs(res.x / se))}, index=names)
        self.summary.loc[names[len(self.covariates):], ['time_ratio', 'lower95', 'upper95']] = np.nan
        self.n_subjects_, self.n_episodes_ = len(self.ids_), len(self._X)
        self.AIC_ = 2 * len(res.x) - 2 * self.loglik_
        return self
