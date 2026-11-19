"""Refit the 9 Oct test model (Weibull, exact time, 'has any' indicators) on exact_v1 (decided rules) and, for
comparison, the 9 Oct build on the same accounts. Writes aft_model_output/fit_v1.pkl and fit_v1_compare.csv.

Run: python refit_test_model_decided_rules.py (no arguments; the two fits run in parallel). Inputs: aft_model_output/exact_v1.parquet,
aft_pilot_test_output/exact.parquet (the 9 Oct build) and sample_998_v4_excl_output/sample_accounts.parquet.
Uses fit(), COUNTS, REF and S from aft_pilot_test/checks/common.py. No truncation correction; every row of the build is
fitted as read (no filter on an 'after' column). The pickle is a list of two result dicts; the csv has coef and SE from each
build and the shift of the v1 coefficient in units of the 9 Oct SE, printed largest first.
"""
import os, sys, pickle, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'aft_pilot_test', 'checks'))
from common import *          # tvaft with the trust-exact fallback, fit, COUNTS, REF, S (also sets HERE)
HERE = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(os.path.dirname(HERE), 'aft_model_output')
ANY = ['visual_media', 'thr_entertainment', 'thr_crypto', 'replies_on_posts']
SOURCES = {'v1_rules': os.path.join(OUT, 'exact_v1.parquet'),
           'oct9_build': os.path.join(os.path.dirname(HERE), 'aft_pilot_test_output', 'exact.parquet')}
KEEP = pd.read_parquet(SOURCES['v1_rules'], columns=['id']).id.unique()   # accounts of the v1 build; both fits use these


def prep(path):
    """Episodes and 9 Oct covariates from one build.

    path: episode parquet. Keeps the KEEP accounts, marks only each account's last episode as the event, adds 'has
    any' for ANY, log(1 + x) counts, username digits and start-year indicators (2014 and earlier as 2015, REF left out).
    Returns (episodes, covariate names, per-account event frame).
    """
    ex = pd.read_parquet(path)
    ex = ex[ex.id.isin(KEEP)].copy()
    ex['event'] = ex.groupby('id').event.transform(lambda e: np.r_[np.zeros(len(e) - 1), 1.0])   # every account accused at its last stop
    acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
    for c in ANY:
        ex[f'any_{c}'] = (ex[c] > 0).astype(float)
    for c in COUNTS:
        ex[c] = np.log1p(ex[c])
    ex['username_digits'] = ex['id'].map(acc['username'].str.contains(r'\d')).astype(float)
    year = pd.to_datetime(ex['id'].map(acc['first_action_utc']), unit='s').dt.year.clip(lower=2015)
    years = sorted(year.unique())
    for y in years:
        if y != REF:
            ex[f'start_{y}'] = (year == y).astype(float)
    covs = (COUNTS + ['flooding', 'has_posted', 'has_commented', 'username_digits'] + [f'start_{y}' for y in years if y != REF]
            + [f'any_{c}' for c in ANY])
    subj = ex.groupby('id', sort=False)['event'].last().to_frame()
    return ex, covs, subj


def one(name):
    """Fit the Weibull model on one build (key of SOURCES); returns its summary, loglik, AIC, convergence, n and the
    fallback log from common."""
    ex, covs, subj = prep(SOURCES[name])
    t = time.time()
    m = fit(ex[['id', 'start', 'stop'] + covs], covs, dist='weibull', subjects=subj)
    print(name, round(m.loglik_, 2), 'n', m.n_subjects_, round(time.time() - t), 's', flush=True)
    return {'name': name, 'summary': m.summary, 'loglik': m.loglik_, 'AIC': m.AIC_, 'conv': m.converged_, 'n': m.n_subjects_,
            'fallback': list(FALLBACKS)}


if __name__ == '__main__':
    path = os.path.join(OUT, 'fit_v1.pkl')
    assert not os.path.exists(path), f'exists, not overwritten: {path}'
    with Pool(2) as pool:
        res = pool.map(one, list(SOURCES), chunksize=1)
    pickle.dump(res, open(path, 'wb'))
    s = {r['name']: r['summary'] for r in res}
    cmp = pd.DataFrame({'coef_oct9': s['oct9_build']['coef'], 'se_oct9': s['oct9_build']['se'],
                        'coef_v1': s['v1_rules']['coef'], 'se_v1': s['v1_rules']['se']})
    cmp['shift_in_se'] = (cmp.coef_v1 - cmp.coef_oct9) / cmp.se_oct9   # how far the decided rules move each coefficient
    cmp.to_csv(os.path.join(OUT, 'fit_v1_compare.csv'))
    print({r['name']: (round(r['loglik'], 2), r['conv']) for r in res})
    print(cmp.round(3).sort_values('shift_in_se', key=abs, ascending=False).head(12).to_string())
