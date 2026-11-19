# r/TheseFuckingAccounts AFT Model and Analysis

> **This is the public copy of a private research repository.** It updates automatically whenever the
> private repository's `main` branch changes. Following the Community Data Science Collective's rules for
> research on Reddit, no data, comment or post text, or Reddit usernames are published here: the four
> case-study accounts appear as `CaseA` to `CaseD`, and other usernames in code comments and tests are
> replaced by made-up examples. Each update is scrubbed and then checked for usernames, data files,
> notebook outputs and secrets before it is pushed; an update that fails any check is not published.
> Changes are made in the private repository, so pull requests here cannot be merged, but you are
> welcome to fork or copy the code.

Dissertation code for a study of the accounts that users of r/TheseFuckingAccounts (TFA) accuse of being bots,
spammers or bought accounts. The project asks **what an account's behaviour on Reddit says about how soon it gets
accused**. It does this in four stages:

1. **Who was accused, and when.** Accused usernames are extracted from every TFA submission (21,469 posts,
   Dec 2014 to Sep 2026), with each account's first-accusation time and how many posts named it.
2. **What those accounts did.** Account status and profiles come from the Reddit API (PRAW), and full post and
   comment histories come from the Arctic Shift archive.
3. **How behaviour relates to time-to-accusation.** An accelerated failure time (AFT) model with time-varying
   covariates measures time from an account's first action to its first accusation. Covariates include activity,
   subreddit groups, links, crossposts, and text copied from other accounts. `exp(coef)` is a time ratio: below 1
   means the behaviour goes with an earlier accusation.
4. **Who they interact with.** Network case studies of single accused accounts and their contacts, plus
   exposure-network models.

Data files are not in this repository. Scripts read and write them on the research server under the same folder
layout (`p2_data_cleaning/data/`, and `*_output/` folders inside `p2_analysis/`). Every script refuses to
overwrite an output that already exists.

## Key definitions

- **Accused account:** named in a TFA submission by one of the extraction rules in
  `p2_data_cleaning/DATA_COLLECTION.md` (helpers, victims and non-usernames excluded).
- **Eligible pool:** accused accounts that are not deleted, have a complete archived history, whose first action
  came after TFA was created and before 30 Nov 2022, and who acted before they were first accused. First
  accusation is the later of the accusing post's creation and its edit (posts edited more than 48 h later are
  dropped). Reddit internal test accounts are excluded.
- **Copied text:** a post or comment of at least 140 characters (links not counted) whose normalised word 3-gram
  shingles have Jaccard similarity of at least 0.7 with an archived text posted earlier by a different author.

## Layout

### `p2_data_cleaning/`: building the account list and collecting histories
- `collect_accused_usernames_and_first_mention.ipynb` extracts accused usernames from TFA posts.
- `collection_scripts/` is the collection pipeline, run in tiers:
  - `build_accused_table.py`, `verify_accused_table.py`, `exclude_helpers_and_victims.py`, `username_shape.py`:
    the accused-account table and the rules that keep helpers, victims and non-usernames out.
  - `tier2_*.py`: account status and profiles from PRAW.
  - `tier3_arctic_account_meta.py`, `tier4_arctic_full_history.py`, `repair_history_cursor.py`,
    `consolidate_account_histories.py`: Arctic Shift account metadata and full post and comment histories.
  - `run_*.sh` chain the tiers; `test_*.py` are paging and throughput tests.
- `collect_sample_2000_full_history.py`, `verify_sample_2000_full_history.py` and
  `collect_case_study_full_history.py` are full-history pulls for samples and case-study accounts.
- `early_notebooks_archive/` holds superseded first drafts, kept for the record.
- `DATA_COLLECTION.md` is the full record of how the collection was run, with counts.

### `p2_analysis/`: samples, variables, models and case studies
- **Samples:** `sample_*.ipynb`, in order: 200 (pilot), then 2,000 and 1,000 stratified samples (v1 to v3), then the
  1,000-account simple random sample (v4), then the 998 left after excluding test accounts.
- **Variables:**
  - `subreddit_*.ipynb` and `subreddit_*guide*.md`: subreddit metadata and subreddit groups (v4 to v10).
  - `link_domain_types_v1.ipynb`: what kind of site each link points to.
  - `collect_crossposts_v4.ipynb`, `collect_*_v4.py`: crossposts, replies, parent authors, comment removal.
  - `shared_text_with_accused_accounts_v1.py`: text shared with other accused accounts.
- **Model:**
  - `tvaft.py`: the time-varying AFT fitter (Weibull, log-normal, log-logistic, generalised gamma; delayed entry;
    robust SEs).
  - `tvaft_simulation.py`, `TV_AFT_validation.ipynb`, `tvaft_eha_check.R`, `eha_aftreg_tv_fix.patch`: its
    validation against lifelines and R `eha`, and Monte Carlo recovery.
  - `aft_pilot_test/`: the first pilot fit.
  - `aft_model/`: the decided rules, the covariate build and every fit (see `aft_model/README.md`).
  - `AFT_pilot_to_full_sample_checklist.md`: what changes between the pilot and the full sample.
- **Copy search:** `copy_search_*` and `lsh_copy_search/`. The exact shingle/Jaccard search for earlier copies of
  each accused text, run on the Hyak cluster (`lsh_copy_search/README.md`).
- **Networks:**
  - `network_case_*.ipynb`: ego networks, alter-alter ties, coordination candidates and identity consistency for
    four case-study accounts.
  - `network_models/`: exposure edges and ERGM inputs.
- `engagement_check_2026-10-13_output/` is a one-off check of subreddit engagement in early October 2026.

## How to run

- **Where:** the collection and analysis scripts run on the research server (Python 3 from Anaconda), from the
  folder each script names (most collection scripts run from `p2_data_cleaning/`). The copy search runs on Hyak
  with Slurm (`sbatch`).
- **Credentials:** PRAW credentials are read from a `.env` file outside the repo and are never written to it.
- **Order:**
  1. Run the collection tiers (`DATA_COLLECTION.md`).
  2. Run `collect_accused_usernames_and_first_mention.ipynb`.
  3. Run the sample notebooks in order.
  4. Build the variables (subreddits, links, crossposts, `collect_*_v4.py`, copy search).
  5. Build and fit with `aft_model/build.py` and the `fit_*.py` scripts.
- **Notebooks:** stored without outputs (`nbstripout`).

## Branches
`main` is the stable branch. Work in progress happens on branches named for what they do (for example
`exact-copy-search`) and is merged into `main` through a pull request.

## Public copy
This repository is private. A scrubbed public copy,
[TheseFuckingAccounts-AFT-Model-Analysis-public](https://github.com/eeporter/TheseFuckingAccounts-AFT-Model-Analysis-public),
is updated automatically on every push to `main` by `.github/workflows/publish-public.yml`. The sync
removes Reddit usernames, never copies private-only files (listed in `.publish/exclude.txt`, e.g. the
audit notes), and blocks the whole update if any username, data file, notebook output or secret is
left. See `.publish/README.md` for how it works and what to do when a sync is blocked.
