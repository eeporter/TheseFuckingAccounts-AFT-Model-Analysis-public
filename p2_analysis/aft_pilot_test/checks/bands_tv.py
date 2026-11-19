"""Do the start-year x log(t) interactions survive once age bands are in (M3)? Also KS for M3."""
import pickle
from common_exact import *
from scipy import stats
from parallel_exact import ks_cens
exec(open('bands.py').read().split('def one')[0].split('FITS =')[0])
b = pickle.load(open('bands.pkl', 'rb'))[0]
cols = COVS + [f'any_{c}' for c in ANY] + BANDS
SY = [c for c in COVS if c.startswith('start_')]
d = ex[['id', 'start', 'stop'] + cols].copy()
for c in SY:
    d[f'{c}_x_logt'] = d[c] * (np.log1p(d.start) - np.log1p(7.0))
P = b['params']; init = np.r_[P[:len(cols)], np.zeros(len(SY)), P[len(cols):]]
m = fit(d, cols + [f'{c}_x_logt' for c in SY], subjects=SUBJ, robust=False, init=init)
LR = 2 * (m.loglik_ - b['loglik'])
print('start-year x log t under M3: LR', round(LR, 1), 'df', len(SY), 'p', stats.chi2.sf(LR, len(SY)), 'conv', m.converged_)
print(m.summary.loc[[f'{c}_x_logt' for c in SY], ['coef', 'se', 'p']].round(3).to_string())
r = b['r']; ev = SUBJ.event.reindex(r.index).to_numpy()
print('KS M3', round(ks_cens(r.to_numpy(), ev), 4))
