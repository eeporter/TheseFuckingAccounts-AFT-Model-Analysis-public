"""M3 = M2 + account-age band indicators (time-varying): a piecewise adjustment of the Weibull clock's pace by age.
Band edges fall on whole days, which are already episode boundaries.

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../../aft_model/); kept for the record.
Bands: days [1, 7), [7, 30), [30, 180), [180, 365), [365, 730), [730, inf); the first day [0, 1) is the reference.
Fits M3 with Weibull and lognormal baselines; for the Weibull fit also keeps Cox-Snell residuals and observed vs
expected accusations per time interval. Inputs: ../exact.parquet, exact_fits.pkl. Output: bands.pkl (list: Weibull
result first, then lognormal). Run from checks/: python bands.py. bands_tv.py execs the setup part of this file."""
import pickle
from multiprocessing import Pool
from common_exact import *
from scipy import stats
ex, acc, COVS, SUBJ = load_exact()
COVS2 = COVS + [f'any_{c}' for c in ANY]
EDGES = [1, 7, 30, 180, 365, 730]
BANDS = [f'age_{a}_{b}' for a, b in zip(EDGES, EDGES[1:] + ['inf'])]
for (a, b), nm in zip(zip(EDGES, EDGES[1:] + [np.inf]), BANDS):
    ex[nm] = ((ex.start >= a) & (ex.start < b)).astype(float)     # band set by the episode start
assert not ((ex.start < 1) & (ex.stop > 1)).any()                  # no episode crosses day 1
FITS = {r['name']: r for r in pickle.load(open(os.path.join(HERE, 'exact_fits.pkl'), 'rb'))}
B = FITS['M2_exact_any']['params']


def one(dist):
    """Fit M3 with baseline dist ('weibull' starts from the M2 estimates with 0 for the bands).

    Returns a dict with loglik, AIC, summary, convergence and params; for Weibull also Cox-Snell residuals 'r' and
    the interval calibration table 'iv' (observed vs expected accusations, O/E, chi-square terms)."""
    cols = COVS2 + BANDS
    init = np.r_[B[:len(COVS2)], np.zeros(len(BANDS)), B[len(COVS2):]] if dist == 'weibull' else None
    m = fit(ex[['id', 'start', 'stop'] + cols], cols, dist=dist, subjects=SUBJ, robust=True, init=init)
    P = m.params_; beta = P[:len(cols)]
    out = {'dist': dist, 'loglik': m.loglik_, 'AIC': m.AIC_, 'summary': m.summary, 'conv': m.converged_, 'params': P}
    if dist == 'weibull':
        r = cox_snell(m); out['r'] = r
        mu, sig = P[-2], np.exp(P[-1]); rate = np.exp(-ex[cols].to_numpy() @ beta)
        endt = ex.groupby('id').stop.last(); ev = SUBJ.event; rows = []
        # expected accusations by day h = sum over accounts of the Weibull cumulative hazard at the clock value
        # Psi(min(end, h)); observed = accused (not censored) accounts with end <= h; as in horizon.py
        for h in [0.25, 1, 2, 7, 14, 30, 90, 180, 365, 730, 2200]:
            mm = ex.start.to_numpy() < h
            psi = pd.Series(rate[mm] * (np.minimum(ex.stop.to_numpy()[mm], h) - ex.start.to_numpy()[mm])).groupby(ex.id.to_numpy()[mm]).sum()
            Eh = np.exp((np.log(psi) - mu) / sig); O = ((endt.reindex(Eh.index) <= h) & ev.reindex(Eh.index).eq(1))
            rows.append({'by_day': h, 'observed': int(O.sum()), 'expected': Eh.sum()})
        c = pd.DataFrame(rows).set_index('by_day'); iv = c.diff().fillna(c)    # cumulative -> per interval
        iv['O/E'] = iv.observed / iv.expected; iv['chi2'] = (iv.observed - iv.expected) ** 2 / iv.expected
        out['iv'] = iv
    return out


if __name__ == '__main__':
    with Pool(2) as pool:
        res = pool.map(one, ['weibull', 'lognormal'])
    pickle.dump(res, open(os.path.join(HERE, 'bands.pkl'), 'wb'))
    for o in res:
        print(o['dist'], 'll', round(o['loglik'], 2), 'AIC', round(o['AIC'], 1), 'conv', o['conv'])
    w = res[0]
    print('LR vs M2', 2 * (w['loglik'] - FITS['M2_exact_any']['loglik']), 'df', len(BANDS))
    print(w['summary'].loc[BANDS].round(3).to_string()); print(w['iv'].round(2).to_string())
    # df = intervals - 2 baseline parameters - the band terms
    print('chi2', w['iv'].chi2.sum().round(1), 'p', stats.chi2.sf(w['iv'].chi2.sum(), len(w['iv']) - 2 - len(BANDS)))
