import pickle, numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width', 220)
R = pickle.load(open('exact_diag.pkl', 'rb'))
F = {r['name']: r for r in pickle.load(open('exact_fits.pkl', 'rb'))}
M2 = F['M2_exact_any']; S = M2['summary']; P = M2['params']
q = pickle.load(open('quick_exact.pkl','rb'))
ks = np.array([r['KS'] for r in R if r['job'] == 'sim'])
print('KS M2', q['KS_M2'], 'boot p', (ks >= q['KS_M2']).mean())
from parallel_exact import ks_cens  # noqa
