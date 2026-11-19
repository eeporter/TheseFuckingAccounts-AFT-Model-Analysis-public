# AFT model build (pilot, decided rules)

The model build that carries the decisions made after the 9 Oct test fit (`../aft_pilot_test/`). Every decided rule is
in `rules.py`, so the larger sample uses the same code. Outputs go to `../aft_model_output/` (git-ignored); scripts refuse
to overwrite an existing output.

| Script | What it does |
|---|---|
| `rules.py` | flooding (10+ threads), visual media by whether it rendered, accounts dropped for a capped archive |
| `build.py` | exact-time covariates for the 9 Oct variable subset with the rules → `exact_v1.parquet` (991 accounts) |
| `fit_v1.py` | Weibull refit on `exact_v1` and on the 9 Oct build for the same 991 accounts → `fit_v1.pkl`, `fit_v1_compare.csv` |
| `checks/visual_rule_counts.py` | pilot link counts under the old host rule vs the render rule |
| `links.py` | link table for any posts/comments, same as `link_domain_types_v1` (reproduces its pre-accusation table exactly) |
| `build.py --to-cutoff` | continues each account past its accusation to 2022-11-30 with archived behaviour → `exact_v1_to_cutoff.parquet` (rows before the accusation identical to `exact_v1`) |
| `truncation.py` | `TruncatedAFT`: likelihood conditioned on T ≤ τ (Lagakos et al. 1988), path after T supplied as episodes |
| `checks/truncation_unit.py` | simulation: per-account log-likelihood matches scipy to 1e-10; corrected fit recovers the truth, uncorrected does not |
| `fit_truncation.py` | correction with τ = cutoff − first action on all 991 (`fit_truncation.pkl`: none, typical, 2015–2018; observed/quiet diverge) |
| `fit_horizon.py` | fixed-horizon design (1 and 2 years), with and without the correction → `fit_horizon_<job>.pkl` |
| `checks/summarise_truncation.py` | all truncation fits side by side → `truncation_summary.csv` |
| `fit_main.py` | **decided sample** (all 11 capped dropped, `exact_v2_to_cutoff`, 1-year window: 757): main, + age bands, and checks → `fit_main_<job>.pkl` |
| `checks/summarise_main.py` | `fit_main` table, age-band LR and calibration → `fit_main_summary.csv` |
| `features.py` | item flags and window statistics for the variables added in build v3/v4 (definitions in its docstring) |
| `build.py --to-cutoff` (v4) | all planned variables (#4, #5, #6b, #7, #10, #14–#18, comment-thread burstiness, replies on comments, #19 removal) → `exact_v4_to_cutoff.parquet`; v2 columns unchanged |
| `checks/v3_bruteforce.py` | naive recount of the new window variables on 40 accounts against the build |
| `fit_full.py` | every planned variable on the decided sample (757): main, no bands, the four #5 forms, profile², flooding 10+, has-any for every count, time × log(t), extended (+ removal) → `fit_full_<job>.pkl` (first run on v3, before comment-thread burstiness: `fit_full_v3_<job>.pkl`) |
| `checks/summarise_full.py` | `fit_full` table, AIC/LR comparison, VIF, calibration → `fit_full_summary.csv`, `fit_full_compare.csv`; results in `full_fit_2026-10-19.md` |
| `build.py --to-cutoff` (v8), `fit_full.py` (2026-10-22) | active-week terms with enough-activity flags, return after silence, title/body length, one Kim & Jo burstiness over posts + thread entries → `exact_v8_to_cutoff.parquet`, `fit_full_<job>.pkl` (+ `no_return`); earlier runs kept as `fit_full_v3_/v4_/v5_<job>.pkl`; results in `full_fit_2026-10-22.md` |

## Right truncation (2026-10-15)

Pool rule 5 (accused before 2022-11-30, kept by the user: pre-ChatGPT) means later cohorts can only enter with short times.
- **Full correction** (−log F(Ψ(τ)), τ = cutoff − first action) needs the covariate path between accusation and cutoff.
  Observed archived path and "quiet" path fail to converge on all 991 (the fit drives F(τ) to ~0 with absurd
  coefficients); the typical-behaviour path converges but has-posted collapses (TR 0.57 → 0.03), as the audit found for
  carry-forward. Not usable on the pilot.
- **Fixed horizon** (recommended, audit agrees): keep accounts whose first action is ≥ H before the cutoff and accused
  within H. Every cohort then has the same window, so start year is no longer tied to how much follow-up an account had.
  It is still a selection (accounts caught within H), common to all cohorts; the results describe timing among accounts
  caught within H, not all accused accounts. Correcting for that selection would again need a post-accusation path.
  H = 1 year: 759 accounts (558 suspended / 201 active), losing 118 accused after a year and 114 who started within a
  year of the cutoff — **all 2022 starters**. Behaviour time ratios are close to the 991-account fit; start-year effects
  remain (2015 0.22, 2019 0.36, 2021 0.60), so they are not only truncation.
- **Checks at H = 1 year**: typical-path correction for the within-H selection moves behaviour TRs by ≤ 1.45 SE (profile threads largest; 757-account fit, `fit_main.py`) (except
  has-posted, 0.57 → 0.05); the observed-path correction converges to a degenerate fit (median F(H) ≈ 0, SEs up to 113)
  and crashes at H = 2 years. **has-posted is unreliable in every corrected fit** — don't interpret it under a correction.
- **Other sensitivity**: H = 2 years (533 accounts) also drops all 2021 starters, so it is not like-for-like with
  H = 1 year (compare only shared cohorts); 2015–2018 starters (weak truncation, 395 accounts).
- **Decided 2026-10-16 (user)**: 1-year window; all 11 capped accounts dropped (`rules.capped_accounts`, build v2).
  Pilot fit: `fit_main.py`. **Age bands included** (user, 2026-10-16): the model is `main_bands` (LR 19.5 / 4 df,
  calibration χ² 30.9 → 14.9); `main` is the no-band comparison.

Inputs: `../sample_998_v4_excl_output/`, `../link_domain_output/links_pre_accusation_v1.parquet`,
`../subreddit_review_output/subreddit_groups_v9.parquet`, `../crosspost_output/crosspost_flags_998.parquet`,
`../replies_output/posts_parts/`, and the 2014–2022 TFA submissions CSV. All planned variables are built as of v4 (2026-10-19).
