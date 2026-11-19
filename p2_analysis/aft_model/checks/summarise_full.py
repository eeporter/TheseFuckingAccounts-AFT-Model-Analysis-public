"""fit_full jobs side by side -> aft_model_output/fit_full_summary.csv (time ratios, robust 95% CIs, p) and
fit_full_compare.csv (n, k, loglik, AIC, LR vs full); refuses to overwrite. Prints the AIC comparison, VIF top, calibration and term tests.

Run: python checks/summarise_full.py (after fit_full.py). Reads every aft_model_output/fit_full_<job>.pkl that exists
for the current fit_full.JOBS (the v9 runs; older fit_full_v*_<job>.pkl files are not read); 'full' must be there.
"""
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
    """time ratio, robust 95% CI and p of one fit's covariates (Weibull mu and log sigma left out)"""
    return f['summary'][['time_ratio', 'lower95', 'upper95', 'p']].drop(['mu', 'log_sigma'])


def save_new(df, name):
    """write df to aft_model_output/<name> as csv; stops if the file exists"""
    path = os.path.join(OUT, name)
    assert not os.path.exists(path), f'exists, not overwritten: {path}'
    df.to_csv(path)


tab = pd.concat({j: tr(f) for j, f in fits.items()}, axis=1)
save_new(tab, 'fit_full_summary.csv')
full = fits['full']
cmp = pd.DataFrame({j: {'n': f['n'], 'k': len(f['covariates']), 'loglik': f['loglik'], 'AIC': f['AIC'], 'conv': f['conv']}
                    for j, f in fits.items()}).T
# every job against full: dAIC < 0 favours the job; LR = 2 (loglik_job - loglik_full) on df = k_job - k_full.
# A job with more terms (df > 0) is tested as is; one with fewer (df < 0, e.g. nobands, t5_none, no_return) is the
# null, so the sign of LR and df is flipped. The chi2 p assumes the smaller model is nested in the larger; for jobs that
# swap terms rather than add or drop them (t5_cat, visual_groups) only the AIC comparison applies. df = 0 gives no p.
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
# calibration: observed vs expected accusations by account age (fit_main.calibration); chi2 summed over the intervals
for j in ['full', 'nobands', 'extended']:
    if j in fits:
        c = fits[j]['calibration']
        print(f'\ncalibration {j}: chi2 {c.chi2.sum():.1f}'); print(c.round(2).to_string())
# n_any counts accounts whose largest value is > 0; for centred terms that means above the active-week mean
print('\naccounts with a nonzero value (full):'); print(pd.Series(full['n_any']).sort_values().to_string())
show = ['time_ratio', 'lower95', 'upper95', 'p']
print('\nfull:'); print(tab['full'][show].round(3).to_string())
