"""Setup for the exact-time refit: same covariates as the test run, plus 'has any' indicators; censoring from exact.parquet.

Part of the superseded 9 Oct 2026 pilot test fit (replaced by ../../aft_model/); kept for the record.
Imported with `from common_exact import *` by the exact-time checks. Reads ../exact.parquet (from ../build_exact.py)
and the pilot sample accounts; writes nothing."""
from common import *          # thread caps, tvaft (with the trust-exact fallback), fit, cox_snell, scores, COUNTS, GROUPS, REF

ANY = ['visual_media', 'thr_entertainment', 'thr_crypto', 'replies_on_posts']   # counts that also get a 0/1 'any' term
_ev = pd.read_parquet(os.path.join(HERE, '..', 'exact.parquet'), columns=['id', 'event']).groupby('id').event.max()
CENSORED = sorted(_ev.index[_ev == 0])   # capped archive ends before accusation (7 in the pilot)


def load_exact():
    """Exact-time episodes with the model covariates.

    Adds any_<c> = 1 when count c > 0 (taken before the log), log(1 + count) for counts, username_digits and
    start-year indicators (2020 = reference). Returns (episodes, accounts, covariate names without the any_ terms,
    subjects table with each account's event from its last episode)."""
    ex = pd.read_parquet(os.path.join(HERE, '..', 'exact.parquet'))
    acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
    for c in ANY:
        ex[f'any_{c}'] = (ex[c] > 0).astype(float)
    for c in COUNTS:
        ex[c] = np.log1p(ex[c])
    ex['username_digits'] = ex['id'].map(acc['username'].str.contains(r'\d')).astype(float)
    year = pd.to_datetime(ex['id'].map(acc['first_action_utc']), unit='s').dt.year.clip(lower=2015)
    years = sorted(year.unique())
    for y in years:
        if y != REF:
            ex[f'start_{y}'] = (year == y).astype(float)
    covs = COUNTS + ['flooding', 'has_posted', 'has_commented', 'username_digits'] + [f'start_{y}' for y in years if y != REF]
    subj = ex.groupby('id', sort=False)['event'].last().to_frame()
    return ex, acc, covs, subj


def sub(ex, subj, keep_ids):
    """Restrict episodes ex and subjects subj to the accounts in keep_ids. Returns (episodes, subjects)."""
    return ex[ex.id.isin(keep_ids)], subj.loc[subj.index.isin(keep_ids)]
