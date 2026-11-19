"""Pilot fits with every planned variable (build v9, exact_v9_to_cutoff.parquet). Earlier runs are kept: v3 (before
comment-thread burstiness) as fit_full_v3_<job>.pkl, v4 (before the active-week terms) as fit_full_v4_<job>.pkl, v5
(active-week terms, before return after silence; full, extended, profile2, t5_cat only) as fit_full_v5_<job>.pkl.
v6 (return after silence) and v7 (title/body length) were stopped before their fits finished, as further changes came in, on the decided sample: all 11 capped
accounts dropped, 1-year window (757 accounts), Weibull, exact time, age bands, robust (sandwich) SEs.

Jobs (python fit_full.py <job>; each writes aft_model_output/fit_full_<job>.pkl, never overwritten):
  full        the main model: 9 Oct set + v3 variables, #5 as log(1 + turns), age bands
  nobands     full without the age bands
  t5_none     #5 left out          | the four pre-agreed forms of #5 (checklist), compared by AIC
  t5_quad     log + log^2          |
  t5_cat      1 / 2-5 / 6+ turns   | (full is the log form)
  profile2    + squared profile-thread term
  flood10     + indicator for the busiest subreddit reaching 10+ threads, next to the share
  anyall      'has any' indicator next to every count (not only the four of the 9 Oct set)
  tlogt       + x * (log(1 + t) - log 8) for flooding, the start years and replies on posts (9 Oct test interactions)
  no_return   full without return after silence (its LR test)
  visual_groups  sensitivity (user, 2026-10-22): visual count and any-visual replaced by indicators for 1, 2-4 and 5+
              visual items in the window (ref: none); descriptive, heavy image posters may be a different kind of account
  extended    full + #19 removal (posts removed by moderators/AutoModerator, comments [removed] today)

Active-week terms (user, 2026-10-22, after audit round 4): commented / posted in the last 7 days are their own terms, so
length, pace, thread length and burstiness describe how the account behaved in the weeks it was active. Pace and
post-hopping need 2+ items (one valid gap) and burstiness 3+ events; each gets an 'enough activity' indicator, and its value is
centred (pace, hopping: log(1 + hours) - log(1 + 24), i.e. at one day) and set to 0 when there is not enough activity,
so the indicator carries the inactive weeks. Thread length and text length stay as they are (0 without a comment/post).
Return after silence (user, 2026-10-22): log(1 + largest gap in days before any item in the window, measured to the
account's previous post or comment, whenever it was); 0 when inactive this week; the first item's gap is 0 (age bands
cover the start). It enters next to the active-week terms and the within-week pace.
Post length (user, 2026-10-22) is split in two: title length (log(1 + mean title characters) over posts in the window,
paired with week_post) and body length (log(1 + mean selftext characters) over self-posts in the window, paired with
week_text_post = any self-post in the window). Comment length stays, paired with week_comment.
Burstiness (user, 2026-10-22) is one measure over posts and comment-thread entries together, the finite-size A_n of
Kim & Jo (2016, Eq. 22), with its own 3+ events flag; it replaces post burstiness and comment-thread burstiness.
Centring (user, 2026-10-23, audit round 4b): each value-when-active term is centred at its duration-weighted mean over
active weeks (pre-accusation episodes of the fitted sample), and set to 0 (= that mean) in weeks its flag is off:
comment length and thread length with commented this week, title length with posted this week, body length with made a
text post this week, pace / hopping / burstiness with their enough-activity flags. This reparameterises the same model
(identical likelihood); each flag then reads as the effect of being active at a typical value, and each value term as
the effect within active weeks. Return after silence is not centred: it is active in any week with a post or a comment,
and the model has no flag for that, so a filler other than 0 would change the model rather than reparameterise it.
Earlier uncentred v8 runs are kept as fit_full_v8u_<job>.pkl, and centred v8 runs (before shared text) as fit_full_v8_<job>.pkl.
Shared text (user, 2026-10-27): log(1 + posts and comments in the window whose text matches an earlier item by a
different accused account), from ../shared_text_v1.py.
The categorical #5 form drops the 1-turn category, which with the others would sum to 'commented this week'.
"""
import os, sys, pickle, time
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(v, '1')
import numpy as np
import pandas as pd

MY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MY)
import rules
import fit_truncation as FT                  # prep, covariates, BANDS; also patches tvaft's fallback optimiser
from fit_horizon import horizon_sample
from fit_main import calibration
from tvaft import TimeVaryingAFT

SOURCE = os.environ.get('AFT_SOURCE', 'exact_v9_to_cutoff.parquet')
V3_COUNTS = ['self_repeats', 'chars_title', 'chars_body', 'chars_comments', 'promo_out_of_place', 'repeat_partners',
             'cross_sub_partners', 'copied', 'shared_text', 'replies_on_comments']
V3_OTHER = ['burstiness_all', 'pace_posts', 'pace_comments', 'hopping', 'has_edited', 'edit_delay']   # already on their scale
REMOVAL = ['removed_posts', 'removed_comments']
WEEK = ['return_gap', 'week_comment', 'week_post', 'week_text_post', 'pace_posts_ok', 'pace_comments_ok', 'hopping_ok', 'burst_ok']
PACE0 = np.log1p(24.0)                     # pace and hopping first centred at one day (re-centred in centre())
CENTRE = {'chars_comments': 'week_comment', 'thread_turns_log': 'week_comment', 'chars_title': 'week_post',
          'chars_body': 'week_text_post', 'pace_posts': 'pace_posts_ok', 'pace_comments': 'pace_comments_ok',
          'hopping': 'hopping_ok', 'burstiness_all': 'burst_ok'}


def centre(ex):
    """centre each value-when-active term at its active-week mean, 0 when its flag is off; returns the means"""
    pre = ex[ex.after == 0]
    w = pre.stop - pre.start
    means = {}
    for v, flag in CENTRE.items():
        on = pre[flag] == 1
        means[v] = float(np.average(pre.loc[on, v], weights=w[on]))
        ex[v] = np.where(ex[flag] == 1, ex[v] - means[v], 0.0)
    return means
LOGT0 = np.log1p(7.0)
JOBS = ['full', 'nobands', 't5_none', 't5_quad', 't5_cat', 'profile2', 'flood10', 'anyall', 'tlogt', 'extended', 'no_return', 'visual_groups']


def prep():
    ex = FT.prep(source=SOURCE)
    for c in V3_COUNTS + REMOVAL:
        ex[c] = np.log1p(ex[c])
    ex['return_gap'] = np.log1p(ex.return_gap)
    ex['week_comment'] = (ex.n_comments_7d > 0).astype(float)
    ex['week_post'] = (ex.n_posts_7d > 0).astype(float)
    ex['week_text_post'] = (ex.n_selfposts_7d > 0).astype(float)
    ok = {'pace_posts': ex.n_posts_7d >= 2, 'pace_comments': ex.n_comments_7d >= 2, 'hopping': ex.hopping < np.log1p(168.0) - 1e-9}
    for c, m in ok.items():
        ex[f'{c}_ok'] = m.astype(float)
        ex[c] = np.where(m, ex[c] - PACE0, 0.0)
    ex['burst_ok'] = (ex.n_burst_7d >= 3).astype(float)
    ex['thread_turns_log'] = np.log1p(ex.thread_turns)
    ex['thread_turns_log2'] = ex.thread_turns_log ** 2
    ex['turns_1'] = (ex.thread_turns == 1).astype(float)
    ex['turns_2_5'] = ex.thread_turns.between(2, 5).astype(float)
    ex['turns_6p'] = (ex.thread_turns >= 6).astype(float)
    ex['thr_profile_sq'] = ex.thr_profile ** 2
    nvis = np.rint(np.expm1(ex.visual_media))                  # back from log(1 + count)
    ex['vis_1'] = (nvis == 1).astype(float)
    ex['vis_2_4'] = nvis.between(2, 4).astype(float)
    ex['vis_5p'] = (nvis >= 5).astype(float)
    ex['flood10'] = (ex.flooding > 0).astype(float)
    for c in FT.COUNTS + V3_COUNTS:
        if f'any_{c}' not in ex and c != 'tfa_posts_30d':
            ex[f'any_{c}'] = (ex[c] > 0).astype(float)
    return ex


def design(ex, job):
    base = FT.covariates(ex, FT.REF, bands=job != 'nobands') + V3_COUNTS + V3_OTHER + WEEK
    t5 = {'t5_none': [], 't5_quad': ['thread_turns_log', 'thread_turns_log2'],
          't5_cat': ['turns_2_5', 'turns_6p']}.get(job, ['thread_turns_log'])
    extra = {'profile2': ['thr_profile_sq'], 'flood10': ['flood10'], 'extended': REMOVAL,
             'anyall': [f'any_{c}' for c in FT.COUNTS + V3_COUNTS if c != 'tfa_posts_30d' and c not in FT.ANY]}.get(job, [])
    if job == 'tlogt':
        names = ['flooding', 'replies_on_posts'] + [c for c in base if c.startswith('start_')]
        for x in names:
            ex[f'{x}_x_logt'] = ex[x] * (np.log1p(ex.start) - LOGT0)
        extra = [f'{x}_x_logt' for x in names]
    covs = [c for c in base + t5 + extra if not (job == 'no_return' and c == 'return_gap')]
    if job == 'visual_groups':
        covs = [c for c in covs if c not in ('visual_media', 'any_visual_media')] + ['vis_1', 'vis_2_4', 'vis_5p']
    return [c for c in covs if ex.loc[ex.after == 0, c].std() > 0]


def vif(ex, covs):
    """duration-weighted VIF of the design (same as the 9 Oct fit and the audit)"""
    pre = ex[ex.after == 0]
    w = (pre.stop - pre.start).to_numpy()
    X = pre[covs].to_numpy(float)
    mu = np.average(X, axis=0, weights=w)
    cv = np.cov((X - mu).T, aweights=w)
    sd = np.sqrt(np.diag(cv))
    return pd.Series(np.diag(np.linalg.pinv(cv / np.outer(sd, sd))), index=covs).sort_values(ascending=False)


def run(job):
    tag = '' if SOURCE.startswith('exact_v9') else SOURCE.split('_')[1] + '_'
    out_path = os.path.join(FT.OUT, f'fit_full_{tag}{job}.pkl')
    assert not os.path.exists(out_path), f'exists, not overwritten: {out_path}'
    t = time.time()
    acc = pd.read_parquet(os.path.join(FT.S, 'sample_accounts.parquet')).set_index('username_key')
    ex = horizon_sample(prep(), acc, rules.HORIZON_DAYS)
    assert set(ex.id) == set(rules.in_window(acc.loc[ex.id.unique()], rules.HORIZON_DAYS))
    means = centre(ex)                         # after the t5 / squared forms, which prep() builds from the raw values
    covs = [c for c in design(ex, job) if c in ex]
    pre = ex[ex.after == 0]
    subj = pre.groupby('id', sort=False)['event'].last().to_frame()
    m = TimeVaryingAFT('weibull').fit(pre[['id', 'start', 'stop'] + covs], subj, covs, robust=True)
    out = {'name': job, 'covariates': covs, 'summary': m.summary, 'loglik': m.loglik_, 'AIC': m.AIC_,
           'conv': m.converged_, 'n': m.n_subjects_, 'params': m.params_, 'cov': m.cov_, 'cov_model': m.cov_model_,
           'secs': time.time() - t, 'vif': vif(ex, covs), 'centre_means': means,
           'n_any': (pre.groupby('id')[covs].max() > 0).sum().to_dict()}
    if job in ('full', 'nobands', 'extended'):
        out['calibration'] = calibration(m, ex, covs)
    pickle.dump(out, open(out_path, 'wb'))
    print(job, 'n', m.n_subjects_, 'k', len(covs), 'loglik', round(m.loglik_, 2), 'AIC', round(m.AIC_, 1),
          'converged', m.converged_, round(time.time() - t), 's', flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
