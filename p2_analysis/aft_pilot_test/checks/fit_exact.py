"""Exact-time refits of the pilot model: M1 (test-run covariates), M2 (M1 + 'has any' indicators), M1 without the 7
censored accounts, and M2 with lognormal and generalized gamma baselines.

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../../aft_model/); kept for the record.
Inputs: ../exact.parquet via common_exact.load_exact(). Output: exact_fits.pkl next to this script, a list of dicts
(summary, loglik, AIC, params, covariances; Cox-Snell residuals for M2_exact_any). Run from checks/: python fit_exact.py."""
import pickle, time
from multiprocessing import Pool
from common_exact import *
ex, acc, COVS, SUBJ = load_exact()
COVS2 = COVS + [f'any_{c}' for c in ANY]
ALL = list(SUBJ.index)
# name -> (covariates, accounts, baseline)
MODELS = {
    'M1_exact': (COVS, ALL, 'weibull'),
    'M2_exact_any': (COVS2, ALL, 'weibull'),
    'M1_drop7': (COVS, [k for k in ALL if k not in CENSORED], 'weibull'),
    'M2_lognormal': (COVS2, ALL, 'lognormal'),
    'M2_gengamma': (COVS2, ALL, 'gengamma'),
}


def one(name):
    """Fit the model `name` from MODELS (robust SEs). Returns a dict of results for exact_fits.pkl."""
    covs, ids, dist = MODELS[name]
    e, s = sub(ex, SUBJ, ids)
    t = time.time()
    m = fit(e[['id', 'start', 'stop'] + covs], covs, dist=dist, subjects=s)
    out = {'name': name, 'summary': m.summary, 'loglik': m.loglik_, 'AIC': m.AIC_, 'conv': m.converged_, 'secs': time.time() - t,
           'n': m.n_subjects_, 'events': int(s.event.sum()), 'params': m.params_, 'cov': m.cov_, 'cov_model': m.cov_model_, 'fallback': list(FALLBACKS)}
    if name.startswith('M2_exact'):
        out['r'] = cox_snell(m); out['ids'] = m.ids_
    print(name, round(m.loglik_, 2), round(time.time() - t), 's', flush=True)
    return out


if __name__ == '__main__':
    with Pool(5) as pool:
        res = pool.map(one, list(MODELS), chunksize=1)
    pickle.dump(res, open(os.path.join(HERE, 'exact_fits.pkl'), 'wb'))
    print('done')
