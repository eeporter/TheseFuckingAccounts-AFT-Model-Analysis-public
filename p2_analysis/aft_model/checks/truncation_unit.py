"""Unit check of TruncatedAFT: on a small simulated data set the per-subject log-likelihood equals an independent
scipy computation of log f(T) - log F(tau) under a piecewise covariate path, and the fit recovers the truth when the
sample is truncated at tau (the uncorrected fit does not)."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truncation import TruncatedAFT
from tvaft import TimeVaryingAFT

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
beta, mu, sig = np.array([0.7, -0.5]), 3.0, 1.2
n = 6000
x1 = rng.binomial(1, 0.5, n).astype(float); x2a = rng.normal(size=n); x2b = rng.normal(size=n); cp = rng.uniform(5, 40, n)
tau = rng.uniform(5, 150, n)
r1 = np.exp(-(beta[0] * x1 + beta[1] * x2a)); r2 = np.exp(-(beta[0] * x1 + beta[1] * x2b))   # clock speed before/after cp
u = np.exp(mu + sig * np.log(-np.log(rng.uniform(size=n))))                                    # baseline clock value at event
T = np.where(u < r1 * cp, u / r1, cp + (u - r1 * cp) / r2)
keep = T < tau
rows = []
for i in np.flatnonzero(keep):
    for s0, s1, x2 in ((0, cp[i], x2a[i]), (cp[i], np.inf, x2b[i])):
        for a0, a1, aft in ((0, T[i], 0), (T[i], tau[i], 1)):
            lo, hi = max(s0, a0), min(s1, a1)
            if hi > lo:
                rows.append((i, lo, hi, x1[i], x2, aft))
ep = pd.DataFrame(rows, columns=['id', 'start', 'stop', 'x1', 'x2', 'after'])
subj = pd.DataFrame({'event': 1.0}, index=np.flatnonzero(keep))

m = TruncatedAFT('weibull')
m.covariates = ['x1', 'x2']; m._prep(ep, subj)
theta = np.r_[beta, mu, np.log(sig)]
ll = np.asarray(m._ll_subjects(theta))
# independent: Weibull log f/F on log scale via scipy gumbel_l
i = m.ids_.to_numpy()
psiT = np.where(T[i] < cp[i], r1[i] * T[i], r1[i] * cp[i] + r2[i] * (T[i] - cp[i]))
psitau = np.where(tau[i] < cp[i], r1[i] * tau[i], r1[i] * cp[i] + r2[i] * (tau[i] - cp[i]))
rate_T = np.where(T[i] < cp[i], r1[i], r2[i])
ref = (stats.gumbel_l.logpdf((np.log(psiT) - mu) / sig) - np.log(sig) - np.log(psiT) + np.log(rate_T)
       - stats.gumbel_l.logcdf(-(np.log(psitau) - mu) / sig * -1) * 0 - np.log(stats.gumbel_l.cdf((np.log(psitau) - mu) / sig)))
print('max |ll - scipy|:', np.abs(ll - ref).max())
fc = TruncatedAFT('weibull').fit(ep, subj, ['x1', 'x2'])
fu = TimeVaryingAFT('weibull').fit(ep[ep.after == 0], subj, ['x1', 'x2'])
print('n kept', keep.sum(), '| truth', theta.round(3))
print('corrected  ', fc.params_.round(3), 'se', np.sqrt(np.diag(fc.cov_)).round(3))
print('uncorrected', fu.params_.round(3))
