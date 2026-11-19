"""Parallel refits: time-varying effects (tv), functional form (ff), sensitivity (sens), parametric bootstrap (sim)."""
import pickle, time
from multiprocessing import Pool
from common import *
from scipy import stats

job = sys.argv[1]
daily, acc, COVS = load()
ep = merge(daily, COVS)
LOGT0 = np.log1p(7.0)               # time-interaction centred at day 7: main effect = time ratio around day 7


def tv_one(c):
    d = daily[['id', 'start', 'stop'] + COVS].copy()
    cols = COVS
    if c is not None:
        names = [c] if c != 'ALL' else COVS
        for x in names:
            d[f'{x}_x_logt'] = d[x] * (np.log1p(d['start']) - LOGT0)
        cols = COVS + [f'{x}_x_logt' for x in names]
    m = fit(d, cols, robust=c not in (None, 'ALL'))
    row = {'cov': c, 'loglik': m.loglik_, 'k': len(m.params_), 'conv': m.converged_}
    if c not in (None, 'ALL'):
        j = cols.index(f'{c}_x_logt')
        se = np.sqrt(m.cov_[j, j])
        row.update(inter=m.params_[j], inter_se_robust=se, main=m.params_[cols.index(c)])
    return row


def ff_one(arg):
    c, kind = arg
    e = ep.copy()
    cols = list(COVS)
    if kind != 'base':
        nm = f'{c}_{kind}'
        e[nm] = e[c] ** 2 if kind == 'sq' else (e[c] > 0).astype(float)
        cols.append(nm)
    b = pickle.load(open(os.path.join(HERE, 'base.pkl'), 'rb'))['params']
    init = b if kind == 'base' else np.r_[b[:len(COVS)], 0.0, b[len(COVS):]]
    t = time.time()
    m = fit(e, cols, robust=kind != 'base', init=init)
    row = {'fallback': list(FALLBACKS), 'secs': time.time() - t, 'cov': c, 'kind': kind, 'loglik': m.loglik_, 'conv': m.converged_}
    if kind != 'base':
        j = cols.index(nm); row.update(extra=m.params_[j], extra_se_robust=np.sqrt(m.cov_[j, j]), main=m.params_[cols.index(c)])
    return row


SENS = {
    'base': (None, COVS),
    'no_tfa_control': (None, [c for c in COVS if c != 'tfa_posts_30d']),
    'no_start_year': (None, [c for c in COVS if not c.startswith('start_')]),
    'no_truncated_histories': (lambda e: ~e.id.isin(TRUNCATED), COVS),
    'no_accused_within_1_day': (lambda e: e.id.map(acc.days_to_accused).ge(1), COVS),
    'suspended_only': (lambda e: e.id.map(acc.account_status).eq('suspended'), COVS),
    'active_only': (lambda e: e.id.map(acc.account_status).eq('active'), COVS),
    'lognormal': (None, COVS),
}


def sens_one(name):
    keep, cols = SENS[name]
    e = ep[keep(ep)] if keep else ep
    m = fit(e, cols, dist='lognormal' if name == 'lognormal' else 'weibull')
    return name, m.summary, m.loglik_, m.n_subjects_


def simulate(params, seed):
    rng = np.random.default_rng(seed)
    p = len(COVS)
    beta, mu, sig = params[:p], params[p], np.exp(params[p + 1])
    last = ep.groupby('id').tail(1)
    ext = last.assign(start=last.stop, stop=last.stop + 1e7)          # covariates carried forward past the observed time
    e = pd.concat([ep, ext]).sort_values(['id', 'start']).reset_index(drop=True)
    rate = np.exp(-e[COVS].to_numpy() @ beta)
    contrib = rate * (e.stop - e.start).to_numpy()
    cum = pd.Series(contrib).groupby(e.id.to_numpy()).cumsum().to_numpy()
    before = cum - contrib
    ids = e.id.unique()
    target = pd.Series(np.exp(mu + sig * np.log(rng.exponential(size=len(ids)))), index=ids)
    tg = e.id.map(target).to_numpy()
    hit = (before < tg) & (tg <= cum)
    tstar = pd.Series(e.start.to_numpy()[hit] + (tg[hit] - before[hit]) / rate[hit], index=e.id.to_numpy()[hit])
    assert len(tstar) == len(ids), 'simulated time beyond extension'
    ts = e.id.map(tstar).to_numpy()
    s = e[e.start.to_numpy() < ts].copy()
    s['stop'] = np.minimum(s.stop.to_numpy(), s.id.map(tstar).to_numpy())
    return s[['id', 'start', 'stop'] + COVS]


def sim_one(seed):
    params = pickle.load(open(os.path.join(HERE, 'sim_truth.pkl'), 'rb'))
    s = simulate(params, seed)
    m = fit(s, COVS, init=params)
    r = cox_snell(m)
    return {'seed': seed, 'params': m.params_, 'se_robust': np.sqrt(np.diag(m.cov_)), 'se_model': np.sqrt(np.diag(m.cov_model_)),
            'KS': stats.kstest(r, 'expon').statistic, 'conv': m.converged_, 'median_T': s.groupby('id').stop.max().median()}


if __name__ == '__main__':
    t = time.time()
    if job == 'tv':
        with Pool(10) as pool:
            res = pool.map(tv_one, [None, 'ALL'] + COVS, chunksize=1)
    elif job == 'ff':
        cnt = [c for c in COVS if c in COUNTS]
        with Pool(23) as pool:
            res = pool.map(ff_one, [(None, 'base')] + [(c, k) for c in cnt for k in ['sq', 'any']], chunksize=1)
    elif job == 'sens':
        with Pool(8) as pool:
            res = pool.map(sens_one, list(SENS), chunksize=1)
    elif job == 'sim':
        m = fit(ep, COVS, robust=False)
        pickle.dump(m.params_, open(os.path.join(HERE, 'sim_truth.pkl'), 'wb'))
        with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as pool:
            res = pool.map(sim_one, range(int(sys.argv[2])), chunksize=1)
    pickle.dump(res, open(os.path.join(HERE, f'{job}.pkl'), 'wb'))
    print(f'{job} done {time.time() - t:.0f}s')
