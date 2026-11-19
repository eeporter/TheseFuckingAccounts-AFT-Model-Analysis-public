"""Estimation checks, goodness of fit, calibration, influence and the far tail for the Weibull test model.

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../../aft_model/); kept for the record.
Sections: A1 optimum (gradient, Hessian), A2 random restarts, A3 daily vs merged episodes, A4 lifelines cross-check,
D1 robust vs model SEs, B Cox-Snell KS / Anderson-Darling and observed/expected by group, far tail, D2 influence.
Inputs: ../daily.parquet via common.load(). Output: base.pkl next to this script (read by horizon.py, figs.py and
parallel.py's ff job), plus printed tables. Run from checks/: python base.py."""
import pickle, time
from common import *
from autograd import grad
from scipy import stats
from lifelines import WeibullAFTFitter

daily, acc, COVS = load()
ep = merge(daily, COVS)
t = time.time(); m = fit(ep, COVS); print(f'base fit {time.time() - t:.0f}s loglik {m.loglik_:.3f}', flush=True)
R = {}
# A1 optimum
# (gradient near 0, and the inverse of the model covariance = observed information must be positive definite)
g = grad(m._negll)(m.params_)
ev = np.linalg.eigvalsh(np.linalg.inv(m.cov_model_))
R['max_abs_gradient'] = float(np.abs(g).max()); R['hessian_min_eig'] = float(ev.min()); R['hessian_cond'] = float(ev.max() / ev.min())
R['converged'] = m.converged_
# A2 restarts
# (refit from perturbed starting values; the same log-likelihood and coefficients mean a single optimum)
rng = np.random.default_rng(11)
R['restart_loglik_diff'] = []
# (a start with betas ~ N(0, 0.5) made BFGS fail and the Nelder-Mead fallback ran > 20 min: noted, not repeated here)
for k in range(4):
    init = m.params_ + rng.normal(0, 0.1 if k < 2 else 0.25, len(m.params_))
    t = time.time(); mm = fit(ep, COVS, robust=False, init=init)
    R.setdefault('restart_seconds', []).append(round(time.time() - t))
    R['restart_loglik_diff'].append(round(mm.loglik_ - m.loglik_, 5))
    R.setdefault('restart_max_coef_diff', []).append(round(float(np.abs(mm.params_ - m.params_).max()), 5))
print(R, flush=True)
# A3 unmerged daily rows give the same likelihood
md = fit(daily[['id', 'start', 'stop'] + COVS], COVS, robust=False, init=m.params_)
R['daily_vs_merged_loglik_diff'] = float(md.loglik_ - m.loglik_); R['daily_vs_merged_max_coef_diff'] = float(np.abs(md.params_ - m.params_).max())
# A4 lifelines cross-check with time-fixed covariates (values on day 0)
# (with constant covariates the model is an ordinary Weibull AFT, so lifelines should give the same fit)
FIX = ['tfa_posts_30d', 'username_digits'] + [c for c in COVS if c.startswith('start_')]
d0 = daily[daily.start.eq(0)].set_index('id')[FIX]
one = d0.assign(start=0.0, stop=acc['days_to_accused'].reindex(d0.index)).reset_index()
mf = fit(one, FIX, robust=False)
lf = WeibullAFTFitter().fit(one.drop(columns=['id', 'start']).assign(E=1), 'stop', 'E')
lc = lf.params_.loc['lambda_'].reindex(FIX)
R['lifelines_max_coef_diff'] = float(np.abs(lc.to_numpy() - mf.params_[:len(FIX)]).max())
R['lifelines_loglik_diff'] = float(lf.log_likelihood_ - mf.loglik_)
# lifelines fits log(rho) with rho = 1 / sigma; tvaft's last parameter is log(sigma)
R['lifelines_sigma_vs_tvaft'] = [float(1 / np.exp(lf.params_.loc['rho_', 'Intercept'])), float(np.exp(mf.params_[-1]))]
print(R, flush=True)

# D1 SE ratio
# (robust / model-based SE; ratios far from 1 point to misspecification)
se_r, se_m = np.sqrt(np.diag(m.cov_)), np.sqrt(np.diag(m.cov_model_))
names = list(m.summary.index)
D1 = pd.DataFrame({'coef': m.params_, 'se_model': se_m, 'se_robust': se_r, 'ratio': se_r / se_m}, index=names)

# B GOF
r = cox_snell(m)
u = 1 - np.exp(-r)
R['KS'] = float(stats.kstest(r, 'expon').statistic)
R['AD'] = float(stats.anderson(r, 'expon').statistic)   # Anderson-Darling (estimated scale; our scale is fixed at 1, so indicative)
R['mean_cs'] = float(r.mean()); R['var_cs'] = float(r.var())     # both 1 for Exp(1)
# calibration O/E by group: sum of residuals = expected events
# lp: each account's time-weighted mean linear predictor x . beta over its follow-up (used for deciles)
lp = pd.Series(ep[COVS].to_numpy() @ m.params_[:len(COVS)] * (ep.stop - ep.start).to_numpy(), index=ep.id).groupby(level=0).sum() / acc['days_to_accused']
info = pd.DataFrame({'r': r, 'status': acc['account_status'], 'start_year': pd.to_datetime(acc['first_action_utc'], unit='s').dt.year.clip(lower=2015),
                     'lp_decile': pd.qcut(lp.reindex(r.index), 10, labels=False) + 1,
                     'truncated': r.index.isin(TRUNCATED), 'n_items': (acc['n_posts'].fillna(0) + acc['n_comments'].fillna(0)).reindex(r.index),
                     'accused_day1': acc['days_to_accused'].reindex(r.index) < 1})
info['activity_tertile'] = pd.qcut(info.n_items.rank(method='first'), 3, labels=['low', 'mid', 'high'])
cal = []
for v in ['status', 'start_year', 'lp_decile', 'activity_tertile', 'truncated', 'accused_day1']:
    for lev, gdf in info.groupby(v, observed=True):
        n, E = len(gdf), gdf.r.sum()        # observed = n (every account had the event); expected = sum of residuals
        p = 2 * min(stats.gamma.cdf(E, n), stats.gamma.sf(E, n))    # sum of n Exp(1) ~ Gamma(n)
        cal.append({'grouping': v, 'level': lev, 'n': n, 'observed': n, 'expected': round(E, 1), 'O/E': round(n / E, 3), 'p': p,
                    'KS': stats.kstest(gdf.r, 'expon').statistic})
cal = pd.DataFrame(cal)
# far tail
# (the 25 accounts with the largest residuals: accused much later than the model expects)
tail = info.sort_values('r', ascending=False).head(25).join(acc[['days_to_accused', 'n_posts', 'n_comments']])
# D2 influence: one-step dfbeta = H^-1 s_i
Sc = scores(m)
dfb = Sc @ m.cov_model_                       # rows = accounts, change in params when the account is dropped is about -dfb
dfbs = pd.DataFrame(dfb / se_r, index=m.ids_, columns=names)     # in units of robust SE
infl = pd.DataFrame({'max_abs_dfbeta_se': dfbs.abs().max(1), 'param': dfbs.abs().idxmax(1)}).sort_values('max_abs_dfbeta_se', ascending=False)
# exact leave-one-out for the 5 most influential
loo = []
for k in infl.index[:5]:
    mk = fit(ep[ep.id != k], COVS, robust=False, init=m.params_)
    j = names.index(infl.loc[k, 'param'])
    loo.append({'account': k, 'param': names[j], 'one_step_change_se': -dfbs.loc[k, names[j]], 'exact_change_se': (mk.params_[j] - m.params_[j]) / se_r[j],
                'tr_full': np.exp(m.params_[j]), 'tr_without': np.exp(mk.params_[j])})
loo = pd.DataFrame(loo)
pickle.dump({'R': R, 'D1': D1, 'cal': cal, 'tail': tail, 'infl': infl, 'dfbs': dfbs, 'loo': loo, 'r': r, 'info': info, 'params': m.params_,
             'cov': m.cov_, 'cov_model': m.cov_model_, 'names': names, 'loglik': m.loglik_, 'lp': lp}, open(os.path.join(HERE, 'base.pkl'), 'wb'))
print(R); print(D1.round(3).to_string()); print(cal.round(3).to_string()); print(tail.to_string()); print(infl.head(15).to_string()); print(loo.round(3).to_string())
