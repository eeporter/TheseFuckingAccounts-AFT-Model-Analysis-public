"""Time ratios of every truncation fit side by side -> aft_model_output/truncation_summary.csv (refuses to overwrite).

Run: python checks/summarise_truncation.py. Reads aft_model_output/fit_truncation.pkl (not fit_truncation_v2.pkl,
which the current fit_truncation.py writes) and every fit_horizon_*.pkl. Fits that failed (no summary) are skipped; the truncation jobs are labelled all991_<job>.
Also writes truncation_summary_fits.csv: n, loglik, convergence and the median F(tau) / F(H) of each corrected fit.
"""
import os, glob, pickle
import pandas as pd
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'aft_model_output')
fits = {}
for k, r in pickle.load(open(os.path.join(OUT, 'fit_truncation.pkl'), 'rb')).items():
    if 'summary' in r:
        fits[f'all991_{k}'] = r
for f in sorted(glob.glob(os.path.join(OUT, 'fit_horizon_*.pkl'))):
    r = pickle.load(open(f, 'rb')); fits[r['name']] = r
cols = {}
for k, r in fits.items():
    cols[(k, 'TR')] = r['summary']['time_ratio']; cols[(k, 'se')] = r['summary']['se']; cols[(k, 'p')] = r['summary']['p']
tab = pd.DataFrame(cols)
# median F: the corrected fits' median probability of accusation by tau (truncation fits) or H (horizon fits)
info = pd.DataFrame({k: {'n': r['n'], 'loglik': r['loglik'], 'converged': r['conv'],
                         'median_F': (r.get('F_tau', r.get('F_H')).median() if ('F_tau' in r or 'F_H' in r) else None)}
                     for k, r in fits.items()}).T
path = os.path.join(OUT, 'truncation_summary.csv')
assert not os.path.exists(path), f'exists, not overwritten: {path}'
tab.to_csv(path); info.to_csv(path.replace('.csv', '_fits.csv'))
print(info.to_string())
