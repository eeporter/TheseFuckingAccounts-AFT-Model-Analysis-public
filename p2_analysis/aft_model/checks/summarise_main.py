"""Table of the fit_main fits -> aft_model_output/fit_main_summary.csv (time ratio, 95% robust CI, p for each fit),
plus the age-band likelihood-ratio test and calibration. Refuses to overwrite.

Run: python checks/summarise_main.py (after all six fit_main.py jobs; reads aft_model_output/fit_main_<job>.pkl).
Also prints, for main_bands and chk_typical, the four coefficients that move most from main, in units of main's SE,
and n, account status counts, loglik and convergence of every fit.
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'aft_model_output')
J = ['main', 'main_bands', 'chk_typical', 'chk_le2020', 'chk_H730', 'chk_early']
R = {j: pickle.load(open(os.path.join(OUT, f'fit_main_{j}.pkl'), 'rb')) for j in J}
tab = pd.concat({j: R[j]['summary'][['time_ratio', 'lower95', 'upper95', 'p']] for j in J}, axis=1).drop(['mu', 'log_sigma'])
path = os.path.join(OUT, 'fit_main_summary.csv')
assert not os.path.exists(path), f'exists, not overwritten: {path}'
tab.to_csv(path)
# main is nested in main_bands (same data, bands added), so LR = 2 * loglik gain on the number of band terms kept
lr = 2 * (R['main_bands']['loglik'] - R['main']['loglik'])
df = len(R['main_bands']['covariates']) - len(R['main']['covariates'])
print(f'age bands: LR {lr:.1f} on {df} df, p = {stats.chi2.sf(lr, df):.4f}')
for j in ['main', 'main_bands']:
    print(j, 'calibration chi2', round(R[j]['calibration'].chi2.sum(), 1))
se = R['main']['summary']['se']
for j in ['main_bands', 'chk_typical']:
    # log TR = coef; terms only in one of the two fits (e.g. the bands) give NaN and are dropped
    d = ((np.log(R[j]['summary'].time_ratio) - np.log(R['main']['summary'].time_ratio)) / se).drop(['mu', 'log_sigma']).dropna().abs()
    print(j, 'largest shifts vs main (SE):', d.sort_values(ascending=False).head(4).round(1).to_dict())
for j in J:
    print(j, R[j]['n'], R[j]['n_status'], 'loglik', round(R[j]['loglik'], 2), 'converged', R[j]['conv'])
