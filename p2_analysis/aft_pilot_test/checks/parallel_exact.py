"""Diagnostics for the exact-time model M2 (exact times + censoring + flooding 2+ + 'has any' for 4 counts), one Pool."""
import pickle, time
from multiprocessing import Pool
from common_exact import *
from scipy import stats

ex, acc, COVS, SUBJ = load_exact()
COVS2 = COVS + [f'any_{c}' for c in ANY]
FITS = {r['name']: r for r in pickle.load(open(os.path.join(HERE, 'exact_fits.pkl'), 'rb'))}
B = FITS['M2_exact_any']['params']
LOGT0 = np.log1p(7.0)
E = ex[['id', 'start', 'stop'] + COVS2]
END = ex.groupby('id', sort=False).stop.last()


def init_with(extra):
    return np.r_[B[:len(COVS2)], np.zeros(extra), B[len(COVS2):]]


def tv(c):
    d = E.copy(); names = COVS2 if c == 'ALL' else [c]
    for x in names:
        d[f'{x}_x_logt'] = d[x] * (np.log1p(d['start']) - LOGT0)
    cols = COVS2 + [f'{x}_x_logt' for x in names]
    m = fit(d, cols, subjects=SUBJ, robust=c != 'ALL', init=init_with(len(names)))
    row = {'job': 'tv', 'cov': c, 'loglik': m.loglik_, 'k': len(m.params_), 'conv': m.converged_}
    if c != 'ALL':
        j = cols.index(f'{c}_x_logt'); row.update(inter=m.params_[j], inter_se=np.sqrt(m.cov_[j, j]), main=m.params_[cols.index(c)])
    return row


def ff(arg):
    c, kind = arg
    d = E.copy(); nm = f'{c}_{kind}'
    d[nm] = d[c] ** 2 if kind == 'sq' else (d[c] > 0).astype(float)
    m = fit(d, COVS2 + [nm], subjects=SUBJ, robust=True, init=init_with(1))
    j = len(COVS2)
    return {'job': 'ff', 'cov': c, 'kind': kind, 'loglik': m.loglik_, 'conv': m.converged_, 'extra': m.params_[j], 'extra_se': np.sqrt(m.cov_[j, j])}


def status(g):
    ids = acc.index[acc.account_status.eq(g)]
    e, s = sub(E, SUBJ, ids)
    m = fit(e, COVS2, subjects=s, robust=True, init=B)
    return {'job': 'status', 'group': g, 'loglik': m.loglik_, 'summary': m.summary, 'conv': m.converged_}


def simulate(seed):
    rng = np.random.default_rng(seed)
    p = len(COVS2); beta, mu, sig = B[:p], B[p], np.exp(B[p + 1])
    last = E.groupby('id', sort=False).tail(1)
    e = pd.concat([E, last.assign(start=last.stop, stop=last.stop + 1e7)]).sort_values(['id', 'start'], kind='stable').reset_index(drop=True)
    rate = np.exp(-e[COVS2].to_numpy() @ beta); contrib = rate * (e.stop - e.start).to_numpy()
    cum = pd.Series(contrib).groupby(e.id.to_numpy()).cumsum().to_numpy(); before = cum - contrib
    ids = e.id.unique()
    tg = e.id.map(pd.Series(np.exp(mu + sig * np.log(rng.exponential(size=len(ids)))), index=ids)).to_numpy()
    hit = (before < tg) & (tg <= cum)
    tstar = pd.Series(e.start.to_numpy()[hit] + (tg[hit] - before[hit]) / rate[hit], index=e.id.to_numpy()[hit])
    cens = END.loc[CENSORED]
    tend = tstar.copy(); ev = pd.Series(1.0, index=tstar.index)
    for k, c in cens.items():                       # same archive cut-off as the real data
        if tstar[k] > c:
            tend[k] = c; ev[k] = 0.0
    te = e.id.map(tend).to_numpy()
    s = e[e.start.to_numpy() < te].copy(); s['stop'] = np.minimum(s.stop.to_numpy(), s.id.map(tend).to_numpy())
    return s, ev.to_frame('event')


def sim(seed):
    s, ev = simulate(seed)
    m = fit(s, COVS2, subjects=ev, robust=False, init=B)
    r = cox_snell(m); evm = ev.event.reindex(r.index).to_numpy()
    return {'job': 'sim', 'seed': seed, 'KS': ks_cens(r.to_numpy(), evm), 'conv': m.converged_}


def ks_cens(r, ev):
    """sup |KM-based CDF of residuals - Exp(1) CDF| (handles the few censored accounts)"""
    o = np.argsort(r); r, ev = r[o], ev[o]; n = len(r)
    S = np.cumprod(1 - ev / (n - np.arange(n)))
    F = 1 - S; Fm = np.r_[0, F[:-1]]; G = 1 - np.exp(-r)
    return float(max(np.abs(F - G).max(), np.abs(Fm - G).max()))


def boot(seed):
    rng = np.random.default_rng(10_000 + seed)
    ids = SUBJ.index.to_numpy(); draw = rng.choice(ids, len(ids), replace=True)
    cnt = pd.Series(draw).value_counts()
    parts, subs = [], []
    g = dict(tuple(E.groupby('id', sort=False)))
    for k, n in cnt.items():
        for j in range(n):
            parts.append(g[k].assign(id=f'{k}#{j}')); subs.append((f'{k}#{j}', SUBJ.loc[k, 'event']))
    d = pd.concat(parts, ignore_index=True)
    s = pd.DataFrame(subs, columns=['id', 'event']).set_index('id')
    m = fit(d, COVS2, subjects=s, robust=False, init=B)
    return {'job': 'boot', 'seed': seed, 'params': m.params_, 'conv': m.converged_}


def run(task):
    kind, arg = task
    t = time.time()
    try:
        out = {'tv': tv, 'ff': ff, 'status': status, 'sim': sim, 'boot': boot}[kind](arg)
    except Exception as e:
        out = {'job': kind, 'arg': arg, 'error': repr(e)}
    out['secs'] = time.time() - t
    print(kind, arg, round(out['secs']), flush=True)
    return out


if __name__ == '__main__':
    counts = [c for c in COVS2 if c in COUNTS]
    tasks = ([('tv', 'ALL')] + [('tv', c) for c in COVS2] + [('ff', (c, 'sq')) for c in counts]
             + [('ff', (c, 'any')) for c in counts if c not in ANY and c != 'tfa_posts_30d']
             + [('status', 'suspended'), ('status', 'active')]
             + [('sim', i) for i in range(int(sys.argv[1]))] + [('boot', i) for i in range(int(sys.argv[2]))])
    print(len(tasks), 'tasks', flush=True)
    with Pool(int(sys.argv[3])) as pool:
        res = pool.map(run, tasks, chunksize=1)
    pickle.dump(res, open(os.path.join(HERE, 'exact_diag.pkl'), 'wb'))
    print('done')
