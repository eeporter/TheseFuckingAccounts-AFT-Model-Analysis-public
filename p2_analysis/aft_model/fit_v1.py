"""Refit the 9 Oct test model (Weibull, exact time, 'has any' indicators) on exact_v1 (decided rules) and, for
comparison, the 9 Oct build on the same accounts. Writes aft_model_output/fit_v1.pkl and fit_v1_compare.csv."""
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
KEEP = pd.read_parquet(SOURCES['v1_rules'], columns=['id']).id.unique()


def prep(path):
    ex = pd.read_parquet(path)
    ex = ex[ex.id.isin(KEEP)].copy()
    ex['event'] = ex.groupby('id').event.transform(lambda e: np.r_[np.zeros(len(e) - 1), 1.0])
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
    cmp['shift_in_se'] = (cmp.coef_v1 - cmp.coef_oct9) / cmp.se_oct9
    cmp.to_csv(os.path.join(OUT, 'fit_v1_compare.csv'))
    print({r['name']: (round(r['loglik'], 2), r['conv']) for r in res})
    print(cmp.round(3).sort_values('shift_in_se', key=abs, ascending=False).head(12).to_string())
