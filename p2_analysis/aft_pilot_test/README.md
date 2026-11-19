# AFT pilot test fit (2026-10-09) and checks

Scripts for the test fit of the time-varying AFT model on the 998-account pilot (`../sample_998_v4_excl_output/`),
run 2026-10-09 in a scratch folder and copied here 2026-10-15 so later work can read and audit them. They use a subset
of the planned variables and are a test of the pipeline, not the final model.

The scripts read and write their data next to themselves (`daily.parquet`, `exact.parquet`, `*.pkl`). The outputs of
the 9 Oct run are in `../aft_pilot_test_output/` (git-ignored); copy them here, or run the builds, before re-running a fit.

| Script | What it does |
|---|---|
| `build.py` | daily covariate grid, 1-day lag → `daily.parquet` |
| `fit.py`, `plots.py` | four baselines by AIC, VIF, figures → `summary_*.csv`, `vif.csv` |
| `build_exact.py` | exact-time covariates, flooding at 2+ threads, censoring of capped accounts → `exact.parquet` |
| `checks/integrity.py` | recounts covariates for 300 random account-days |
| `checks/base.py`, `parallel.py`, `horizon.py` | daily-model checks: KS bootstrap, calibration, time interactions, functional form, status split, simulation, influence |
| `checks/fit_exact.py`, `parallel_exact.py`, `quick_exact.py`, `summ_exact*.py` | exact-time refits and the same checks |
| `checks/bands.py`, `bands_tv.py` | age-band baseline indicators; start-year × log t with bands |
| `checks/figs.py` | check figures → `p2_figures/aft_test_run_2026-10-09/checks/` |
| `busiest.py` | threads per 7 days in the busiest subreddit (2026-10-15; set the flooding cut-off at 10) |

Findings and open items are in `../AFT_pilot_to_full_sample_checklist.md`. Flooding was later set to 10+ threads in the
busiest subreddit (2026-10-15); `build_exact.py` still has the 2+ trial rule.
The decided rules (flooding 10+, visual media by rendering, capped accounts dropped) are applied in `../aft_model/`.
