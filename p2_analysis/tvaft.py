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

In the TFA study a subject is one account, time is days since its first archived action, the event is its first
accusation, and an episode is a stretch between two of its actions over which the moving 7-day-window covariates are
constant. Because each subject is one account, the robust (sandwich) covariance is clustered by account.

Library module, reads and writes no files. Use:
    from tvaft import TimeVaryingAFT
    m = TimeVaryingAFT('weibull').fit(episodes, subjects, covariates, robust=True)
    m.summary         # coef, se, time_ratio = exp(coef), 95% CI, p; one row per covariate plus baseline parameters
Callers: aft_model/fit_*.py, aft_model/truncation.py, TV_AFT_validation.ipynb (with tvaft_simulation.py and
tvaft_eha_check.R).
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
    """Log density and log survival of the baseline clock value u > 0; theta = (mu, log_sigma[, q]).

    u is on the time scale (days in this study), not log scale; arrays are elementwise. Returns (log f0(u), log S0(u)),
    where f0 is the density of U itself (not of log U). Written with autograd.numpy so it can be differentiated.
    """
    y = anp.log(u)
    mu, sig = theta[0], anp.exp(theta[1])           # sigma is fitted on the log scale so it stays positive
    w = (y - mu) / sig                              # standardized log clock: log U = mu + sigma * W
    # Each logf below is log f_W(w) - log(sigma) - log(u): the change of variables from W to U = exp(mu + sigma W)
    # has Jacobian dw/du = 1 / (sigma * u).
    if dist == 'weibull':
        logS = -anp.exp(w)                          # S_W(w) = exp(-e^w)
        logf = w - anp.exp(w) - anp.log(sig) - y    # f_W(w) = exp(w - e^w)
    elif dist == 'lognormal':
        logS = norm.logcdf(-w)                      # S_W(w) = Phi(-w), computed on log scale for accuracy in the tail
        logf = norm.logpdf(w) - anp.log(sig) - y
    elif dist == 'loglogistic':
        logS = -anp.logaddexp(0.0, w)               # S_W(w) = 1 / (1 + e^w); logaddexp avoids overflow for large w
        logf = w - 2 * anp.logaddexp(0.0, w) - anp.log(sig) - y   # f_W(w) = e^w / (1 + e^w)^2
    elif dist == 'gengamma':
        # Prentice's parameterisation: with k = q^-2, z = k * exp(q w) is Gamma(k, 1). q = 0 itself is not handled here
        # (k would be infinite); the fit starts at q = 0.5.
        q = theta[2]
        k = q ** -2
        z = k * anp.exp(q * w)
        logf = anp.log(anp.abs(q)) - anp.log(sig) - y - gammaln(k) + k * anp.log(k) + k * (q * w - anp.exp(q * w))
        # z increases with w when q > 0, so S is the upper regularized incomplete gamma; when q < 0 z decreases with w
        # and S is the lower one. 1e-300 keeps the log finite when S underflows to 0.
        logS = anp.where(q > 0, anp.log(gammaincc(k, z) + 1e-300), anp.log(gammainc(k, z) + 1e-300))
    else:
        raise ValueError(f'dist must be one of {DISTRIBUTIONS}')
    return logf, logS


def split_at(episodes, times):
    """Split each subject's episode that straddles times[id] (e.g. the entry time) into two, so Psi at that time is exact.

    episodes: DataFrame with id, start, stop (+ covariates, copied to both halves). times: Series or dict id -> time, in
    the same units as start/stop. Returns a new DataFrame sorted by id, start; episodes not straddling are unchanged.
    """
    t = episodes['id'].map(times)
    straddle = (episodes['start'] < t) & (episodes['stop'] > t)   # strictly inside; a boundary already at t is left alone
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

    Parameter vector: beta (one per covariate, in the order of `covariates`), then mu, log_sigma[, q].
    Attributes after fit: params_, loglik_, cov_ (robust if requested), cov_model_ (inverse Hessian), summary, AIC_,
    converged_, ids_ (subject order), T_ (event/censoring time per subject), n_subjects_, n_episodes_.
    """

    def __init__(self, dist='weibull'):
        """dist: baseline distribution, one of DISTRIBUTIONS."""
        if dist not in DISTRIBUTIONS:
            raise ValueError(f'dist must be one of {DISTRIBUTIONS}')
        self.dist = dist

    def _prep(self, episodes, subjects):
        """Validate the episode table and cache the arrays the likelihood needs.

        Sets _X (episode x covariate matrix), _dur (episode lengths), _first_idx/_last_idx (row of each subject's first
        and last episode), _event, _pre_entry (1 for episodes that end at or before the subject's entry time),
        _has_entry, ids_ and T_. Subjects are ordered as their ids first appear after sorting episodes by id, start.
        """
        ep = episodes.sort_values(['id', 'start']).reset_index(drop=True)
        if set(ep['id']) != set(subjects.index):
            raise ValueError('episodes and subjects must cover the same ids')
        if not (ep['stop'] > ep['start']).all():
            raise ValueError('every episode needs stop > start')
        nxt = ep.groupby('id')['start'].shift(-1)          # next episode's start within the subject; NaN on the last
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
        self._last_idx = np.r_[self._first_idx[1:] - 1, len(ep) - 1]   # last row of a subject = row before the next first
        self._event = sub['event'].to_numpy(float)
        entry = sub['entry'].to_numpy(float) if 'entry' in sub else np.zeros(len(sub))
        ep_entry = np.repeat(entry, self._last_idx - self._first_idx + 1)   # each subject's entry time on each of its rows
        if 'entry' in sub:
            # Psi(entry) is summed over whole episodes, so it is exact only if an episode ends at the entry time
            stops_at_entry = pd.Series(np.isclose(ep['stop'].to_numpy(), ep_entry)).groupby(ep['id'].to_numpy()).any()
            if not stops_at_entry.loc[order[entry > 0]].all():
                raise ValueError('an episode must end exactly at each entry time; use split_at(episodes, entry)')
        self._pre_entry = (ep['stop'].to_numpy(float) <= ep_entry + 1e-9).astype(float)   # 1e-9: float tolerance
        self._has_entry = entry > 0
        self.ids_ = order
        self.T_ = ep['stop'].to_numpy()[self._last_idx]

    def _psi(self, contrib):
        """Per-subject sum of the per-episode values in contrib (length n_episodes) -> array of length n_subjects.

        With contrib = exp(-x_j . beta) * duration_j this is the accumulated clock Psi at each subject's last stop.
        A running total is used (sum for a subject = total at its last row minus total before its first row) because
        autograd can differentiate it, unlike an in-place groupby.
        """
        cs = anp.cumsum(contrib)
        before = anp.concatenate([anp.zeros(1), cs[:-1]])     # running total up to, not including, each row
        return cs[self._last_idx] - before[self._first_idx]

    def _ll_subjects(self, params):
        """Log-likelihood contribution of each subject (array of length n_subjects) at params = (beta, theta)."""
        p = len(self.covariates)
        beta, theta = params[:p], params[p:]
        contrib = anp.exp(-anp.dot(self._X, beta)) * self._dur   # clock time accumulated in each episode
        logf, logS = baseline_logf_logS(self.dist, self._psi(contrib), theta)
        # Event: density of T is f0(Psi(T)) * dPsi/dT, and dPsi/dT = exp(-x_last . beta) at T (the Jacobian of the time
        # change, using the covariates of the last episode). Censored: survival S0(Psi(T)).
        ll = self._event * (logf - anp.dot(self._X[self._last_idx], beta)) + (1 - self._event) * logS
        if self._has_entry.any():
            # left truncation: condition on surviving to entry, i.e. subtract log S0(Psi(entry)), with Psi(entry) summed
            # over the episodes before entry. Subjects without entry get a dummy clock of 1 (their Psi(entry) is 0, whose
            # log is -inf) and their term is then zeroed.
            psi_e = self._psi(contrib * self._pre_entry)
            _, logS_e = baseline_logf_logS(self.dist, anp.where(self._has_entry, psi_e, 1.0), theta)
            ll = ll - anp.where(self._has_entry, logS_e, 0.0)
        return ll

    def _negll(self, params):
        """Total negative log-likelihood, the objective minimised by fit."""
        return -anp.sum(self._ll_subjects(params))

    def fit(self, episodes, subjects, covariates, robust=False, init=None):
        """Fit by maximum likelihood and build self.summary; returns self.

        episodes, subjects, robust: see the class docstring (times in any consistent unit; days in this study).
        covariates: list of covariate column names in episodes. init: optional starting parameter vector
        (beta..., mu, log_sigma[, q]); by default beta = 0, mu = mean log T, log_sigma = log sd of log T, q = 0.5.
        """
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
        self.converged_ = bool(res.success)     # success of the last optimiser run (Nelder-Mead if BFGS failed)
        self.params_, self.loglik_ = res.x, -res.fun
        H = hessian(self._negll)(res.x)         # observed information (Hessian of the negative log-likelihood)
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
            # sandwich H^-1 (sum_i s_i s_i') H^-1, with one score row s_i per subject (summed over all its episodes),
            # so episodes of the same subject (account) are treated as a cluster; no small-sample correction
            self.cov_ = H_inv @ (scores.T @ scores) @ H_inv
        else:
            self.cov_ = H_inv
        self.robust = robust
        se = np.sqrt(np.diag(self.cov_))
        names = self.covariates + ['mu', 'log_sigma', 'q'][:n_theta]
        # Wald 95% intervals on the coefficient scale, exponentiated to time ratios
        self.summary = pd.DataFrame({'coef': res.x, 'se': se, 'time_ratio': np.exp(res.x),
                                     'lower95': np.exp(res.x - 1.96 * se), 'upper95': np.exp(res.x + 1.96 * se),
                                     'p': 2 * stats.norm.sf(np.abs(res.x / se))}, index=names)
        self.summary.loc[names[len(self.covariates):], ['time_ratio', 'lower95', 'upper95']] = np.nan   # not time ratios
        self.n_subjects_, self.n_episodes_ = len(self.ids_), len(self._X)
        self.AIC_ = 2 * len(res.x) - 2 * self.loglik_
        return self
