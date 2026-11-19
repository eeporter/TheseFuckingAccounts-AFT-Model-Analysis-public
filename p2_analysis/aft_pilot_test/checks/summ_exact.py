import pickle, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
pd.set_option('display.width', 200)
R = pickle.load(open('exact_diag.pkl', 'rb'))
F = {r['name']: r for r in pickle.load(open('exact_fits.pkl', 'rb'))}
M2 = F['M2_exact_any']; ll0 = M2['loglik']
err = [r for r in R if 'error' in r]; print('errors', len(err), err[:3])
nc = [ (r['job'], r.get('cov', r.get('seed', r.get('group')))) for r in R if 'conv' in r and not r['conv']]; print('not converged', nc)
tv = [r for r in R if r['job'] == 'tv' and 'error' not in r]
a = [r for r in tv if r['cov'] == 'ALL'][0]; df = a['k'] - len(M2['params'])
print('tv ALL LR', round(2*(a['loglik']-ll0),1), 'df', df, 'p', stats.chi2.sf(2*(a['loglik']-ll0), df))
t = pd.DataFrame([r for r in tv if r['cov'] != 'ALL']).set_index('cov')
t['LR'] = 2*(t.loglik-ll0); t['p'] = stats.chi2.sf(t.LR.clip(lower=0), 1); t['holm'] = multipletests(t.p, method='holm')[1]
t['z'] = t.inter/t.inter_se
print(t.sort_values('p')[['main','inter','inter_se','z','LR','p','holm']].head(12).round(4).to_string())
ff = pd.DataFrame([r for r in R if r['job'] == 'ff' and 'error' not in r])
ff['LR'] = 2*(ff.loglik-ll0); ff['p'] = stats.chi2.sf(ff.LR.clip(lower=0), 1); ff['holm'] = multipletests(ff.p, method='holm')[1]
print(ff.sort_values('p')[['cov','kind','extra','extra_se','LR','p','holm']].head(12).round(4).to_string())
st = {r['group']: r for r in R if r['job'] == 'status'}
LR = 2*(st['suspended']['loglik']+st['active']['loglik']-ll0); k = len(M2['params'])
print('status LR', round(LR,1), 'df', k, 'p', stats.chi2.sf(LR, k))
ks = np.array([r['KS'] for r in R if r['job'] == 'sim' and 'KS' in r]); print('sim n', len(ks), 'q95', np.quantile(ks,.95).round(4), 'median', np.median(ks).round(4))
qk = pickle.load(open('quick_exact.pkl','rb')); print(type(qk), list(qk.keys()) if isinstance(qk, dict) else '')
