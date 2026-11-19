"""Figures for the daily-model checks: calibration over time, KS bootstrap, simulation bias and coverage, time-varying
effects, sensitivity refits, and Cox-Snell residuals by account status.

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../../aft_model/); kept for the record.
Inputs (opened by relative path): base.pkl, horizon.pkl, sim.pkl, sim_truth.pkl, tv.pkl, sens.pkl.
Outputs: c1_*.png .. c6_*.png in p2_figures/aft_test_run_2026-10-09/checks/ of the main checkout.
Run from checks/: python figs.py (after base.py, parallel.py tv/sens/sim and horizon.py)."""
import pickle
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from scipy import stats
FIG = '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_figures/aft_test_run_2026-10-09/checks'
os.makedirs(FIG, exist_ok=True)
daily, acc, COVS = load()
b = pickle.load(open('base.pkl', 'rb')); hz = pickle.load(open('horizon.pkl', 'rb'))
names = b['names']

# 1 calibration over time
iv = hz['iv'].copy()
lab = ['<6h', '6h–1d', '1–2d', '2–7d', '1–2w', '2w–1m', '1–3m', '3–6m', '6–12m', '1–2y', '2y+']    # intervals between horizon.py's horizons
fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(iv))
ax.bar(x - 0.2, iv.observed, 0.4, label='observed accusations', color='tab:orange')
ax.bar(x + 0.2, iv.expected, 0.4, label='expected by the model', color='tab:blue', alpha=0.7)
ax.errorbar(x + 0.2, iv.expected, yerr=1.96 * np.sqrt(iv.expected), fmt='none', ecolor='k', lw=1)   # Poisson 95% band
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel('accounts'); ax.legend()
ax.set_title('When accusations happen: observed vs what the Weibull model expects (bars ±95%)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'c1_calibration_time.png'), dpi=130); plt.close(fig)

# 2 KS null distribution
KS = hz['KS_sims']
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(KS, 30, color='grey', alpha=0.7, label='300 data sets simulated from the fitted model')
ax.axvline(b['R']['KS'], color='tab:red', lw=2, label=f"real data ({b['R']['KS']:.3f}); p = {hz['ks_boot_p']:.3f}")
# 1.36 / sqrt(n) is the 5% KS critical value for a fully specified null; too lenient when parameters are estimated
ax.axvline(1.36 / np.sqrt(998), color='k', ls='--', lw=1, label='textbook cut-off 0.043 (too lenient)')
ax.set_xlabel('KS distance of Cox-Snell residuals'); ax.legend(fontsize=8); ax.set_title('Goodness of fit, with estimated parameters')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'c2_ks_bootstrap.png'), dpi=130); plt.close(fig)

# 3 simulation recovery
sim = pickle.load(open('sim.pkl', 'rb')); truth = pickle.load(open('sim_truth.pkl', 'rb'))
P = np.array([r['params'] for r in sim]); SR = np.array([r['se_robust'] for r in sim]); SM = np.array([r['se_model'] for r in sim])
# coverage: share of simulations whose 95% CI (estimate +/- 1.96 SE) contains the true value
cov_r = (np.abs(P - truth) <= 1.96 * SR).mean(0); cov_m = (np.abs(P - truth) <= 1.96 * SM).mean(0)
bias = (P.mean(0) - truth) / P.std(0)
k = len(COVS)
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
y = np.arange(k)
axes[0].barh(y, bias[:k], color='tab:blue'); axes[0].axvline(0, color='k', lw=1)
axes[0].set_yticks(y); axes[0].set_yticklabels(COVS); axes[0].set_xlim(-0.5, 0.5); axes[0].set_xlabel('bias (in standard deviations)'); axes[0].set_title('Bias: near 0 = estimates recover the truth')
axes[1].plot(cov_r[:k], y, 'o', label='robust SE'); axes[1].plot(cov_m[:k], y, 's', mfc='none', label='model SE')
axes[1].axvline(0.95, color='k', ls='--', lw=1); axes[1].axvspan(0.925, 0.975, color='grey', alpha=0.15)
axes[1].set_xlabel('95% CI coverage'); axes[1].legend(); axes[1].set_title('Coverage: should be ≈ 0.95 (grey = simulation noise)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'c3_simulation.png'), dpi=130); plt.close(fig)

# 4 time-varying effects
# (LR test of each single interaction against the base fit; plot the TR path for the 6 smallest p-values)
tv = pd.DataFrame(pickle.load(open('tv.pkl', 'rb')))
t = tv[tv['cov'].notna() & tv['cov'].ne('ALL')].set_index('cov')
base_ll = tv[tv['cov'].isna()].loglik.iloc[0]
t['p'] = stats.chi2.sf(2 * (t.loglik - base_ll), 1)
show = t.sort_values('p').head(6)
days = np.array([0.5, 1, 2, 7, 30, 90, 365, 1000])
fig, ax = plt.subplots(figsize=(8, 4.5))
for c, r in show.iterrows():
    tr = np.exp(r.main + r.inter * (np.log1p(days) - np.log1p(7)))      # time ratio at day t: exp(main + inter * (log(1+t) - log 8))
    ax.plot(days, tr, marker='o', ms=3, label=f'{c} (p={r.p:.1e})')
ax.axhline(1, color='k', lw=1, ls='--'); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('days since first activity'); ax.set_ylabel('time ratio'); ax.legend(fontsize=8)
ax.set_title('Effects that change over time (other variables: no evidence they do)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'c4_time_varying_effects.png'), dpi=130); plt.close(fig)

# 5 sensitivity
s = pickle.load(open('sens.pkl', 'rb'))
TR = pd.DataFrame({n: sm.time_ratio for n, sm, ll, ns in s}).loc[COVS]
LO = pd.DataFrame({n: sm.lower95 for n, sm, ll, ns in s}).loc[COVS]; HI = pd.DataFrame({n: sm.upper95 for n, sm, ll, ns in s}).loc[COVS]
cols = ['base', 'no_truncated_histories', 'no_accused_within_1_day', 'suspended_only', 'active_only', 'lognormal']
clr = ['k', 'tab:purple', 'tab:red', 'tab:orange', 'tab:green', 'tab:grey']
vars_ = [c for c in COVS if not c.startswith('start_')]
fig, ax = plt.subplots(figsize=(9, 11))
for j, (c, col) in enumerate(zip(cols, clr)):
    yy = np.arange(len(vars_)) + (j - 2.5) * 0.12          # offset each refit vertically around the variable's row
    ax.errorbar(TR.loc[vars_, c], yy, xerr=[TR.loc[vars_, c] - LO.loc[vars_, c], HI.loc[vars_, c] - TR.loc[vars_, c]], fmt='o', ms=3, color=col, lw=0.8, label=c)
ax.axvline(1, color='k', ls='--', lw=1); ax.set_xscale('log'); ax.set_xlim(0.1, 10)
ax.set_yticks(np.arange(len(vars_))); ax.set_yticklabels(vars_); ax.invert_yaxis(); ax.legend(fontsize=8, loc='lower right')
ax.set_xlabel('time ratio (log scale)'); ax.set_title('Sensitivity refits')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'c5_sensitivity.png'), dpi=130); plt.close(fig)

# 6 Cox-Snell by status
r = b['r']; st = acc['account_status'].reindex(r.index)
fig, ax = plt.subplots(figsize=(6, 4.5))
for g, col in [('suspended', 'tab:red'), ('active', 'tab:green')]:
    rr = np.sort(r[st == g]); n = len(rr); H = np.cumsum(1 / (n - np.arange(n)))    # Nelson-Aalen of the residuals
    ax.plot(rr, H, '.', ms=3, color=col, label=f'{g} (n={n})')
ax.plot([0, 6], [0, 6], 'k--', lw=1); ax.set_xlim(0, 6); ax.set_ylim(0, 6)
ax.set_xlabel('Cox-Snell residual'); ax.set_ylabel('cumulative hazard'); ax.legend(); ax.set_title('Fit by status (status is not in the model)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'c6_cox_snell_status.png'), dpi=130); plt.close(fig)
print('saved', sorted(os.listdir(FIG)))
