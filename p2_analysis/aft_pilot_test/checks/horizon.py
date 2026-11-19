"""Calibration over time: observed accusations by day h vs the model's expected count sum_i H_i(min(T_i, h)) (martingale identity),
overall and by status; plus the KS p-value from the parametric bootstrap.

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../../aft_model/); kept for the record.
Inputs: ../daily.parquet via common.load(), base.pkl (base.py) and sim.pkl (parallel.py sim). Output: horizon.pkl
(read by figs.py) and printed tables. Run from checks/ (the pickles are opened by relative path): python horizon.py."""
import pickle
from common import *
from scipy import stats
daily, acc, COVS = load()
p = pickle.load(open('base.pkl', 'rb'))
beta, mu, sig = p['params'][:len(COVS)], p['params'][-2], np.exp(p['params'][-1])
d = daily[['id', 'start', 'stop'] + COVS].copy()
d['rate'] = np.exp(-d[COVS].to_numpy() @ beta)             # clock speed on each day
H = lambda psi: np.exp((np.log(np.maximum(psi, 1e-300)) - mu) / sig)    # Weibull cumulative hazard of the clock value psi
rows = []
status = acc['account_status']
for h in [0.25, 1, 2, 7, 14, 30, 90, 180, 365, 730, 2200]:
    seg = d[d.start < h]
    psi = (seg.rate * (np.minimum(seg.stop, h) - seg.start)).groupby(seg.id).sum()   # clock at min(T, h)
    E = H(psi); T = acc['days_to_accused'].reindex(E.index)
    O = (T <= h)
    for grp, mask in [('all', slice(None)), ('suspended', status.reindex(E.index).eq('suspended')), ('active', status.reindex(E.index).eq('active'))]:
        o, e = O[mask].sum(), E[mask].sum()
        rows.append({'by_day': h, 'group': grp, 'observed': int(o), 'expected': round(e, 1), 'O/E': round(o / e, 3),
                     'z': round((o - e) / np.sqrt(e), 2)})       # Poisson-style z
cal = pd.DataFrame(rows)
print(cal.pivot(index='by_day', columns='group', values=['observed', 'expected', 'O/E']).to_string())
# interval version (events in (h_prev, h] vs expected in that interval) for 'all'
a = cal[cal.group.eq('all')].set_index('by_day')
iv = pd.DataFrame({'observed': a.observed.diff().fillna(a.observed), 'expected': a.expected.diff().fillna(a.expected)})
iv['O/E'] = (iv.observed / iv.expected).round(3); iv['chi2'] = (iv.observed - iv.expected) ** 2 / iv.expected
print(iv.round(2).to_string())
# df = intervals - 2 (for the two baseline parameters)
print('interval chi-square', iv.chi2.sum().round(1), 'df', len(iv) - 2, 'p', stats.chi2.sf(iv.chi2.sum(), len(iv) - 2))
# bootstrap p: share of data sets simulated from the fitted model whose KS is at least the real one
sim = pickle.load(open('sim.pkl', 'rb'))
KS = np.array([r['KS'] for r in sim])
print('observed KS', round(p['R']['KS'], 4), '| bootstrap p =', (KS >= p['R']['KS']).mean(), '| sim 95th', np.quantile(KS, .95).round(4))
pickle.dump({'cal_time': cal, 'iv': iv, 'ks_boot_p': (KS >= p['R']['KS']).mean(), 'KS_sims': KS}, open('horizon.pkl', 'wb'))
