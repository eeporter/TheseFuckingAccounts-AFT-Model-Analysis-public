"""Diagnostics for the exact-time model M2 (exact times + censoring + flooding 2+ + 'has any' for 4 counts), one Pool.

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../../aft_model/); kept for the record.
Tasks: tv (x * (log(1 + t) - log 8) per covariate and all at once), ff (x^2 and 1[x > 0] per count), status (separate
fits for suspended and active accounts), sim (parametric bootstrap of the censoring-aware KS statistic), boot
(account-level nonparametric bootstrap of the coefficients). All start from the M2_exact_any estimates.
Inputs: ../exact.parquet, exact_fits.pkl (fit_exact.py). Output: exact_diag.pkl next to this script.
Run from checks/: python parallel_exact.py <n sim> <n boot> <workers>. bands_tv.py and summ_exact2.py import ks_cens
from here, which also runs the module-level setup below."""
import pickle, time
from multiprocessing import Pool
from common_exact import *
from scipy import stats

ex, acc, COVS, SUBJ = load_exact()
COVS2 = COVS + [f'any_{c}' for c in ANY]
FITS = {r['name']: r for r in pickle.load(open(os.path.join(HERE, 'exact_fits.pkl'), 'rb'))}
B = FITS['M2_exact_any']['params']
LOGT0 = np.log1p(7.0)                       # interactions centred at day 7, as in parallel.py
E = ex[['id', 'start', 'stop'] + COVS2]
END = ex.groupby('id', sort=False).stop.last()   # end of follow-up (accusation or archive end)


def init_with(extra):
    """M2 estimates with `extra` zeros inserted after the coefficients (starting values for M2 + extra terms)."""
    return np.r_[B[:len(COVS2)], np.zeros(extra), B[len(COVS2):]]


def tv(c):
    """Constant-TR check: refit M2 with covariate x log-time interactions.

    c: one covariate name or 'ALL'. Returns a dict with loglik, number of parameters, convergence and, for one
    covariate, the interaction, its robust SE and the main effect."""
    d = E.copy(); names = COVS2 if c == 'ALL' else [c]
    for x in names:
        d[f'{x}_x_logt'] = d[x] * (np.log1p(d['start']) - LOGT0)    # log t taken at the episode start
    cols = COVS2 + [f'{x}_x_logt' for x in names]
    m = fit(d, cols, subjects=SUBJ, robust=c != 'ALL', init=init_with(len(names)))
    row = {'job': 'tv', 'cov': c, 'loglik': m.loglik_, 'k': len(m.params_), 'conv': m.converged_}
    if c != 'ALL':
        j = cols.index(f'{c}_x_logt'); row.update(inter=m.params_[j], inter_se=np.sqrt(m.cov_[j, j]), main=m.params_[cols.index(c)])
    return row


def ff(arg):
    """Functional-form check: refit M2 with x^2 ('sq') or 1[x > 0] ('any') for one count.

    arg: (covariate, kind). Returns a dict with loglik, convergence and the extra term with its robust SE."""
    c, kind = arg
    d = E.copy(); nm = f'{c}_{kind}'
    d[nm] = d[c] ** 2 if kind == 'sq' else (d[c] > 0).astype(float)
    m = fit(d, COVS2 + [nm], subjects=SUBJ, robust=True, init=init_with(1))
    j = len(COVS2)
    return {'job': 'ff', 'cov': c, 'kind': kind, 'loglik': m.loglik_, 'conv': m.converged_, 'extra': m.params_[j], 'extra_se': np.sqrt(m.cov_[j, j])}


def status(g):
    """Refit M2 on accounts with account_status g ('suspended' or 'active'). Returns loglik and summary."""
    ids = acc.index[acc.account_status.eq(g)]
    e, s = sub(E, SUBJ, ids)
    m = fit(e, COVS2, subjects=s, robust=True, init=B)
    return {'job': 'status', 'group': g, 'loglik': m.loglik_, 'summary': m.summary, 'conv': m.converged_}


def simulate(seed):
    """Simulate event times from the fitted M2 (Weibull) along each account's real covariate path, then censor the
    censored accounts at their real archive end if the simulated time is later.

    seed: RNG seed. Returns (episodes cut at the simulated end, subjects table with event)."""
    rng = np.random.default_rng(seed)
    p = len(COVS2); beta, mu, sig = B[:p], B[p], np.exp(B[p + 1])
    last = E.groupby('id', sort=False).tail(1)
    # carry the last covariates forward far past the observed end, so every simulated time falls inside the data
    e = pd.concat([E, last.assign(start=last.stop, stop=last.stop + 1e7)]).sort_values(['id', 'start'], kind='stable').reset_index(drop=True)
    rate = np.exp(-e[COVS2].to_numpy() @ beta); contrib = rate * (e.stop - e.start).to_numpy()
    cum = pd.Series(contrib).groupby(e.id.to_numpy()).cumsum().to_numpy(); before = cum - contrib   # clock at episode end / start
    ids = e.id.unique()
    # baseline clock target U = exp(mu + sigma * log E), E ~ Exp(1), i.e. Weibull
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
    """One parametric-bootstrap draw: simulate, refit M2, and return the KS statistic of its Cox-Snell residuals."""
    s, ev = simulate(seed)
    m = fit(s, COVS2, subjects=ev, robust=False, init=B)
    r = cox_snell(m); evm = ev.event.reindex(r.index).to_numpy()
    return {'job': 'sim', 'seed': seed, 'KS': ks_cens(r.to_numpy(), evm), 'conv': m.converged_}


def ks_cens(r, ev):
    """sup |KM-based CDF of residuals - Exp(1) CDF| (handles the few censored accounts)

    r: Cox-Snell residuals; ev: event indicators (0 = censored). Returns the KS distance as a float."""
    o = np.argsort(r); r, ev = r[o], ev[o]; n = len(r)
    S = np.cumprod(1 - ev / (n - np.arange(n)))                  # Kaplan-Meier survival of the residuals
    F = 1 - S; Fm = np.r_[0, F[:-1]]; G = 1 - np.exp(-r)         # Fm = KM CDF just before each residual
    return float(max(np.abs(F - G).max(), np.abs(Fm - G).max()))


def boot(seed):
    """One nonparametric bootstrap refit: resample accounts with replacement (each copy gets its own id, suffix
    '#j'), refit M2 with model SEs. Returns the parameter vector and convergence."""
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
    """Pool worker: run one (kind, arg) task, catching errors so one failure does not stop the pool.

    Returns the task's result dict (or an error record) with its run time in seconds."""
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
    # 'any' terms only for counts that do not already have one in M2 (and not for the TFA control)
    tasks = ([('tv', 'ALL')] + [('tv', c) for c in COVS2] + [('ff', (c, 'sq')) for c in counts]
             + [('ff', (c, 'any')) for c in counts if c not in ANY and c != 'tfa_posts_30d']
             + [('status', 'suspended'), ('status', 'active')]
             + [('sim', i) for i in range(int(sys.argv[1]))] + [('boot', i) for i in range(int(sys.argv[2]))])
    print(len(tasks), 'tasks', flush=True)
    with Pool(int(sys.argv[3])) as pool:
        res = pool.map(run, tasks, chunksize=1)
    pickle.dump(res, open(os.path.join(HERE, 'exact_diag.pkl'), 'wb'))
    print('done')
