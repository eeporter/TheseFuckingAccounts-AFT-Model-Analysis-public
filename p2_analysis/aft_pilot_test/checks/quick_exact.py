import pickle
from common_exact import *
from scipy import stats
ex, acc, COVS, SUBJ = load_exact()
COVS2 = COVS + [f'any_{c}' for c in ANY]
FITS = {r['name']: r for r in pickle.load(open(os.path.join(HERE, 'exact_fits.pkl'), 'rb'))}
out = {}
for name, covs in [('M1_exact', COVS), ('M2_exact_any', COVS2)]:
    P = FITS[name]['params']; beta, mu, sig = P[:len(covs)], P[-2], np.exp(P[-1])
    rate = np.exp(-ex[covs].to_numpy() @ beta)
    H = lambda psi: np.exp((np.log(np.maximum(psi, 1e-300)) - mu) / sig)
    rows = []
    endt = ex.groupby('id').stop.last(); ev = SUBJ.event
    for h in [0.25, 1, 2, 7, 14, 30, 90, 180, 365, 730, 2200]:
        m = ex.start.to_numpy() < h
        psi = pd.Series(rate[m] * (np.minimum(ex.stop.to_numpy()[m], h) - ex.start.to_numpy()[m])).groupby(ex.id.to_numpy()[m]).sum()
        Eh = H(psi); O = ((endt.reindex(Eh.index) <= h) & ev.reindex(Eh.index).eq(1))
        rows.append({'by_day': h, 'observed': int(O.sum()), 'expected': Eh.sum()})
    c = pd.DataFrame(rows).set_index('by_day')
    iv = c.diff().fillna(c); iv['O/E'] = iv.observed / iv.expected; iv['chi2'] = (iv.observed - iv.expected) ** 2 / iv.expected
    out[name] = iv
    print(name); print(iv.round(2).to_string()); print('chi2', iv.chi2.sum().round(1), 'p', stats.chi2.sf(iv.chi2.sum(), len(iv) - 2))
r = FITS['M2_exact_any']['r']; evr = SUBJ.event.reindex(r.index).to_numpy()
exec(open('parallel_exact.py').read().split('def boot')[0].split('def ks_cens')[1].join(['def ks_cens', '']) if False else '')
def ks_cens(r, ev):
    o = np.argsort(r); r, ev = r[o], ev[o]; n = len(r)
    S = np.cumprod(1 - ev / (n - np.arange(n))); F = 1 - S; Fm = np.r_[0, F[:-1]]; G = 1 - np.exp(-r)
    return float(max(np.abs(F - G).max(), np.abs(Fm - G).max()))
out['KS_M2'] = ks_cens(r.to_numpy(), evr); print('KS M2', out['KS_M2'])
st = acc.account_status.reindex(r.index)
for g in ['suspended', 'active']:
    m = (st == g).to_numpy(); print(g, 'O', evr[m].sum(), 'E', r[m].sum().round(1), 'O/E', (evr[m].sum() / r[m].sum()).round(3))
# VIF duration-weighted
w = (ex.stop - ex.start).to_numpy(); X = ex[COVS2].to_numpy(float)
mu_ = (w[:, None] * X).sum(0) / w.sum(); Xc = X - mu_; cv = (w[:, None] * Xc).T @ Xc / w.sum(); sd = np.sqrt(np.diag(cv))
vif = pd.Series(np.diag(np.linalg.inv(cv / np.outer(sd, sd))), index=COVS2).sort_values(ascending=False)
print('VIF top'); print(vif.head(10).round(2).to_string()); out['vif'] = vif
out['r_M2'] = r
pickle.dump(out, open(os.path.join(HERE, 'quick_exact.pkl'), 'wb'))
