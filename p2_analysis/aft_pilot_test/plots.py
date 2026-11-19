"""Fit diagnostics for the AFT test run: Cox-Snell residuals, baseline QQ, time-ratio forest plot, TFA control."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_analysis')
import tvaft
from tvaft import TimeVaryingAFT

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_figures/aft_test_run_2026-10-09'
os.makedirs(FIG, exist_ok=True)

# rebuild the episodes exactly as fit.py does
src = open(os.path.join(HERE, 'fit.py')).read().split('res = {}')[0]
exec(src)

DISTS = ['weibull', 'lognormal', 'loglogistic', 'gengamma']
fits, resid = {}, {}
for dist in DISTS:
    m = TimeVaryingAFT(dist).fit(ep, subjects, COVS)
    p = len(COVS)
    beta, theta = m.params_[:p], m.params_[p:]
    psi = np.asarray(m._psi(np.exp(-m._X @ beta) * m._dur))
    _, logS = tvaft.baseline_logf_logS(dist, psi, theta)
    fits[dist], resid[dist] = m, -np.asarray(logS)      # Cox-Snell residual: Exp(1) if the model is right
    print(dist, 'AIC', round(m.AIC_, 1))

# 1. Cox-Snell residuals: all accounts had the event, so compare with Exp(1) directly (cumulative hazard plot)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
for ax, dist in zip(axes, DISTS):
    r = np.sort(resid[dist]); n = len(r)
    H = np.cumsum(1 / (n - np.arange(n)))                # Nelson-Aalen cumulative hazard of the residuals
    ax.plot(r, H, '.', ms=3)
    lim = max(r.max(), H.max())
    ax.plot([0, 6], [0, 6], 'k--', lw=1)
    ax.set_xlim(0, 6); ax.set_ylim(0, 6)
    ax.set_title(f'{dist}  (AIC {fits[dist].AIC_:.0f})')
    ax.set_xlabel('Cox-Snell residual')
axes[0].set_ylabel('cumulative hazard of residuals')
fig.suptitle('Cox-Snell residuals — points on the dashed line = good fit', y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, '1_cox_snell.png'), dpi=130, bbox_inches='tight'); plt.close(fig)

# 2. QQ: fitted baseline clock vs its assumed distribution (log scale)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, dist in zip(axes, DISTS):
    u = np.sort(1 - np.exp(-resid[dist]))                # fitted CDF value of each account's accumulated clock
    pp = (np.arange(len(u)) + 0.5) / len(u)
    ax.plot(pp, u, '.', ms=3); ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ks = stats.kstest(u, 'uniform')
    ax.set_title(f'{dist}  (KS D = {ks.statistic:.3f})'); ax.set_xlabel('expected probability')
axes[0].set_ylabel('fitted probability')
fig.suptitle('Probability plot of the fitted model — points on the dashed line = good fit', y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, '2_probability_plot.png'), dpi=130, bbox_inches='tight'); plt.close(fig)

# 3. Forest plot of time ratios (Weibull, robust SEs from fit.py)
s = pd.read_csv(os.path.join(HERE, 'summary_weibull.csv'), index_col=0).loc[COVS]
s = s.iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 11))
y = np.arange(len(s))
col = np.where(s.upper95 < 1, 'tab:red', np.where(s.lower95 > 1, 'tab:blue', 'grey'))
ax.errorbar(s.time_ratio, y, xerr=[s.time_ratio - s.lower95, s.upper95 - s.time_ratio], fmt='none', ecolor=col, lw=1.5)
ax.scatter(s.time_ratio, y, c=col, zorder=3)
ax.axvline(1, color='k', lw=1, ls='--'); ax.set_xscale('log')
ax.set_yticks(y); ax.set_yticklabels(s.index)
ax.set_xlabel('time ratio (log scale)   <1 = accused sooner   >1 = accused later')
ax.set_title('Weibull test run: time ratios with 95% robust CIs')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '3_time_ratios.png'), dpi=130); plt.close(fig)

# 4. TFA control: TFA posts in the previous 30 days over calendar time, with the pilot's accusation dates
tfa_t = pd.to_datetime(tfa_sorted := np.sort(pd.read_csv('/data/Emily_data/TheseFuckingAccounts_submissions.csv',
                       usecols=['created_utc']).created_utc.pipe(pd.to_numeric, errors='coerce').dropna().to_numpy()), unit='s')
daily_n = pd.Series(1, index=tfa_t).resample('D').sum()
roll = daily_n.rolling(30).sum()
acc_dates = pd.to_datetime(acc['first_mention_adj_utc'], unit='s')
fig, ax1 = plt.subplots(figsize=(12, 4.5))
ax1.plot(roll.index, roll.values, color='tab:purple', lw=1)
ax1.set_ylabel('TFA posts in previous 30 days', color='tab:purple')
ax2 = ax1.twinx()
ax2.hist(acc_dates, bins=pd.date_range('2015-01-01', '2022-12-01', freq='MS'), color='tab:orange', alpha=0.45)
ax2.set_ylabel('pilot accusations per month', color='tab:orange')
ax1.set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2022-12-01'))
ax1.set_title('TFA activity (control) and when the pilot accounts were accused')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '4_tfa_activity.png'), dpi=130); plt.close(fig)

# 5. Observed time to accusation (log scale) by start year
d = acc[['days_to_accused', 'first_action_utc']].copy()
d['start_year'] = pd.to_datetime(d.first_action_utc, unit='s').dt.year.clip(lower=2015)
fig, ax = plt.subplots(figsize=(9, 4.5))
yrs = sorted(d.start_year.unique())
ax.boxplot([np.log10(d.loc[d.start_year == yv, 'days_to_accused']) for yv in yrs], labels=yrs)
ax.set_yticks(np.log10([0.01, 0.1, 1, 10, 100, 1000])); ax.set_yticklabels(['0.01', '0.1', '1', '10', '100', '1000'])
ax.set_ylabel('days to first accusation (log)'); ax.set_xlabel('year of first archived activity')
ax.set_title('Time to accusation by start year (raw data)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '5_time_by_start_year.png'), dpi=130); plt.close(fig)

print('TFA 30-day count over pilot timelines:', ep.assign(n=np.expm1(ep.tfa_posts_30d)).n.describe().round(0).to_dict())
print(d.groupby('start_year').days_to_accused.median().round(1).to_dict())
print('saved to', FIG)
