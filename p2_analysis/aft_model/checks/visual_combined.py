"""Visual media (user, 2026-10-22): count and 'any visual media' stay in the model but are reported as one combined
effect at 1, 2, 5 and 10 items in the 7-day window vs none, log TR = b_count * log(1 + k) + b_any, with a 95% CI from
the joint robust covariance. Usage: python checks/visual_combined.py [job ...] (default: full extended)."""
import os, sys, pickle
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'aft_model_output')
rows = []
for job in sys.argv[1:] or ['full', 'extended']:
    f = pickle.load(open(os.path.join(OUT, f'fit_full_{job}.pkl'), 'rb'))
    covs = f['covariates']
    i, j = covs.index('visual_media'), covs.index('any_visual_media')
    b, V = f['params'][[i, j]], f['cov'][np.ix_([i, j], [i, j])]
    for k in [1, 2, 5, 10]:
        x = np.array([np.log1p(k), 1.0])
        est, se = x @ b, np.sqrt(x @ V @ x)
        rows.append({'job': job, 'items': k, 'time_ratio': np.exp(est), 'lower95': np.exp(est - 1.96 * se),
                     'upper95': np.exp(est + 1.96 * se), 'p': 2 * (1 - __import__('scipy').stats.norm.cdf(abs(est / se)))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
