"""fit_full jobs side by side -> aft_model_output/fit_full_summary.csv (time ratios, robust 95% CIs, p) and
fit_full_compare.csv (n, k, loglik, AIC, LR vs full); refuses to overwrite. Prints the AIC comparison, VIF top, calibration and term tests."""
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(HERE), 'aft_model_output')
sys.path.insert(0, HERE)
from fit_full import JOBS

fits = {j: pickle.load(open(os.path.join(OUT, f'fit_full_{j}.pkl'), 'rb')) for j in JOBS
        if os.path.exists(os.path.join(OUT, f'fit_full_{j}.pkl'))}


def tr(f):
    return f['summary'][['time_ratio', 'lower95', 'upper95', 'p']].drop(['mu', 'log_sigma'])


def save_new(df, name):
    path = os.path.join(OUT, name)
    assert not os.path.exists(path), f'exists, not overwritten: {path}'
    df.to_csv(path)


tab = pd.concat({j: tr(f) for j, f in fits.items()}, axis=1)
save_new(tab, 'fit_full_summary.csv')
full = fits['full']
cmp = pd.DataFrame({j: {'n': f['n'], 'k': len(f['covariates']), 'loglik': f['loglik'], 'AIC': f['AIC'], 'conv': f['conv']}
                    for j, f in fits.items()}).T
cmp['dAIC_vs_full'] = cmp.AIC - full['AIC']
cmp['LR_vs_full'] = 2 * (cmp.loglik - full['loglik'])
cmp['df'] = cmp.k - len(full['covariates'])
cmp['p_LR'] = [stats.chi2.sf(lr, df) if df > 0 else (stats.chi2.sf(-lr, -df) if df < 0 else np.nan)
               for lr, df in zip(cmp.LR_vs_full, cmp.df)]
save_new(cmp, 'fit_full_compare.csv')
pd.set_option('display.width', 200)
print(cmp.round(3).to_string())
print('\nVIF top (full):'); print(full['vif'].head(15).round(2).to_string())
print('\nVIF > 5 (extended):'); print(fits['extended']['vif'][fits['extended']['vif'] > 5].round(2).to_string()) if 'extended' in fits else None
for j in ['full', 'nobands', 'extended']:
    if j in fits:
        c = fits[j]['calibration']
        print(f'\ncalibration {j}: chi2 {c.chi2.sum():.1f}'); print(c.round(2).to_string())
print('\naccounts with a nonzero value (full):'); print(pd.Series(full['n_any']).sort_values().to_string())
show = ['time_ratio', 'lower95', 'upper95', 'p']
print('\nfull:'); print(tab['full'][show].round(3).to_string())
