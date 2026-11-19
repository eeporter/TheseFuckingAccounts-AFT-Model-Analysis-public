# AFT model: pilot (v4, 998 accounts) → larger sample checklist

The pilot sample (`sample_998_v4_excl_output/`: the v4 random draw of 1,000 minus 2 Reddit internal test accounts) is a
**pilot** for drafting the time-varying AFT model. The final analysis will use a larger sample. Every item below was
decided or sized on the pilot and must be **re-run on the larger sample and compared** — report both where they differ.

## How the pilot was selected (draw the larger sample the same way)

First mention = the **later** of the first accusing post's `created_utc` and its `edited` time (the TFA CSV stores post
text as retrieved, including edits, and only the last edit time is known). Only each account's first accusing post is stored.

| Step | Rule | Pool after |
|---|---|---|
| 1 | Accused and not deleted (PRAW status active or suspended) — `accused_accounts_not_deleted.parquet` | 61,931 |
| 2 | Complete Arctic Shift history (`arctic_pull_done.parquet`, capped 10k pulls count as complete) | 55,082 |
| 3 | First archived action after TFA was created (2014-12-21 00:03:16 UTC, `1419120196`); first action = earliest of Arctic Shift `earliest_post_at`, `earliest_comment_at` and the oldest stored post/comment | 52,964 |
| 4 | First accusing post not edited more than 48 hours after it was created | 47,748 |
| 5 | First mention (edit-adjusted) before 2022-11-30 00:00 UTC (`1669766400`) | 40,279 |
| 6 | First action before 2022-11-30 | 40,267 |
| 7 | First action before first mention (time to accusation > 0) | 38,346 |
| 8 | Not a Reddit internal test account: username does not start `lss_mobile`, `lss_web`, `lss_bvt` or `lssqa` (added 2026-10-07; these post almost only in Reddit's test subreddits) | **38,334** |
| Draw | 1,000 at random, **not stratified by status**, `eligible.sample(n=1000, random_state=2026)`, drawn before rule 8; the 2 rule-8 accounts were then dropped without replacement | 713 suspended / 285 active (998) |

Built by `sample_1000_v4_random_eligible.ipynb` then `sample_998_v4_exclusions.ipynb` (rule 8). For the larger sample, apply
rule 8 to the pool **before** drawing. History of the rules: `Sample_500_suspended_500_active_eligible*.ipynb`
v1–v3). Time zero = first recorded (archived) activity, not account creation. Status is not a model variable.

## Specification choices to compare (pilot vs larger sample)
- [ ] **#5 comment thread length** (longest back-and-forth on one post in the last 7 days): compare the four pre-agreed
      versions — none, log(1 + max), log + log², categories — by AIC (likelihood-ratio test for curvature), inside the full
      model. Pilot category cut-offs **1 / 2–5 / 6+** were set because only 153 pilot accounts reach 2+ and 21 reach 21+;
      with more accounts, finer categories (e.g. a separate 21+) may become estimable.
- [ ] **Baseline distribution**: Weibull vs log-normal vs log-logistic vs generalized gamma by AIC (pilot, no covariates:
      active ≈ Weibull, suspended ≈ log-normal).
- [ ] **Overlap / VIF** among the time-varying variables (VIF > 5 rule) — correlations may change with more accounts.
- [ ] **#16 link types outside their home groups**: pilot counts are thin for some (news links outside news/politics:
      18 accounts; YouTube outside home: 62). Recheck counts and whether each is estimable; classify new domains.
- [ ] **#6 edits** (both kept, 2026-10-08: edits in the last 7 days counted at edit time, and log median time to edit +
      has-edited indicator): pilot has only 76/998 accounts with a pre-accusation edit — recheck the count is enough.
- [ ] **Window length**: pilot uses 7 days only; check whether a 24-hour window adds anything with more data.
- [ ] **Model checks from the pilot test run (2026-10-09, page "AFT Pilot Model Checks")** — repeat on the real model and
      on the larger sample: KS of Cox-Snell residuals judged against a parametric bootstrap (pilot Weibull p = 0.017; the
      textbook 1.36/√n cut-off is too lenient), calibration over time bands (pilot: excess at 6h–1d, lull at 6–12 months,
      excess after 1 year), time × log(t) interactions (pilot: flooding, start 2015/2019, replies on posts change over
      time), "has any" indicators next to log counts (pilot: visual media, entertainment, crypto, replies need them),
      status heterogeneity (pooled vs split rejected), robust-SE coverage for rare variables (pilot 85–89%) → bootstrap CIs.
- [ ] **Truncated histories**: 11 pilot accounts hit the 10,000-item cap (oldest first); for 7 the archive ends before the
      accusation (the other 4 have complete pre-accusation history). Trial 2026-10-09: censored at their archive end —
      dropping them instead moved estimates by at most 1.2 SE (downvoted comments). **Decided 2026-10-15 (user): drop
      them**; 2026-10-16 (user): drop **all 11** capped accounts, including the 4 with complete pre-accusation history — the 7 whose capped archive ends before the accusation (6 active, 1 suspended; includes the 5 skipped in
      reply collection), rule `rules.capped_before_accusation` in `aft_model/`. Pilot model sample 991 (712 suspended /
      279 active). On the larger sample: recount, and use Hyak full histories if they become available.
- [ ] **Flooding definition** — decided 2026-10-15 (user): flooding needs **10+ threads in the busiest subreddit within the
      last 7 days**; the value is then the busiest subreddit's share of the window's threads, 0 otherwise (replaces the
      2+ threads trial). Pilot: 135/998 accounts ever reach 10 threads in one subreddit in a week (85 counting posts only;
      128 of the 991 kept); typical active week median 1, busiest week median 2. Refit 2026-10-15 (`aft_model/refit_test_model_decided_rules.py`,
      9 Oct variables): flooding time ratio 0.73 (2+ rule) → 1.17, SE 0.21 → 0.36, not significant; other estimates
      moved ≤ 0.4 SE. Recheck the count and the cut-off on the larger sample.
- [ ] **Exact-time covariates** (trial 2026-10-09): values change at each item's time and 7 days later instead of daily;
      removes the day-0 artefact (174 pilot accounts accused within day 0). Log-likelihood −4,673 → −4,512.
- [ ] **Age-band baseline** — **included, user 2026-10-16** (1-year window: four bands 1/7/30/180 days, LR 19.5 on 4 df,
      calibration χ² 30.9 → 14.9, behaviour TRs move ≤ 0.6 SE). Trial 2026-10-09: pilot accusations dip at 6–12 months of account age (38 vs 89 expected
      under the Weibull) across start-year cohorts. Six time-varying age-band indicators (1/7/30/180/365/730 days):
      LR 66.8 on 6 df, calibration χ² 60 → 8, main time ratios unchanged. Recheck the dip on the larger sample.
- [ ] **Start-year × time interactions** survive the age bands on the pilot (LR 71 on 7 df). Check whether the sample
      rule "accused before 2022-11-30" (right truncation: later cohorts can only enter with short times) explains part
      of this before the larger draw. **2026-10-15**: user keeps the 2022-11-30 cutoff (pre-ChatGPT). Full truncation
      correction is unstable on the pilot (`aft_model/README.md`); **decided 2026-10-16 (user): fixed 1-year window** (757 accounts after dropping all 11 capped; audit agrees; drops all 2022
      starters and is itself a selection on being caught within a year) — start-year effects persist under it. Recheck both on the larger sample (more accounts may make the correction estimable).
- [ ] **Robust vs bootstrap SEs** (pilot exact model): bootstrap SEs 0–28% larger (median 6%); only has_commented changes
      significance. Report bootstrap CIs.

## Subreddit groups
- [ ] **New subreddits**: the larger sample will use subreddits the pilot never saw. Rerun the pipeline
      (`Subreddit_metadata` → `Subreddit_groups` with the same guides: `subreddit_classification_guide.md`,
      `subreddit_classification_guide_without_bio.md`, `subreddit_entertainment_guide.md`) and carry over the `MANUAL` overrides.
- [ ] **Group viability** (rule of thumb ≥ ~30 accounts with any activity): on the pilot, mental health/support (46 accounts,
      one account = 93% of activity), kids'-content spam (1 account), tech, identity, sports-only, hobbies were left in
      `other`. Recheck — some may become viable groups.
- [ ] **Concentrated groups**: news (one account = 86% of activity), sales (58%), karma farming (top 3 = 88%) on the pilot.
- [ ] Adult rule for banned/private subreddits (≥ 90% NSFW archived posts) — recheck agreement with Reddit's label.
- [ ] **Banned/private subreddits relabelled from archived bios** (`subreddit_groups_v10_banned_private_from_archived_bios.ipynb`, 2026-10-09): Arctic Shift's
      subreddit dump (`/data/reddit/subreddits_2025-01.zst`, or a newer dump) holds bios for almost all of them. Rerun the
      dump extract for the new subreddits, apply the adult rule (dump `over18` or ≥ 90% NSFW archived posts), and label the
      rest with `subreddit_classification_guide_archived_bio.md`; `unclear` keeps the content-based label. Pilot: 730 sent to
      agents, 53 unclear. Bio labels describe what a subreddit says it is for, the earlier labels what the sampled
      accounts posted there — the two disagree most for spam-heavy subreddits (e.g. homework-help and VPN subreddits:
      content said sales, bio said other).

## Link types (#16)
- [ ] **New domains**: rerun `link_domain_types_v1.ipynb` on the larger sample and label domains not in
      `link_domain_llm_labels_v1.csv` with `link_domain_classification_guide.md`; carry over `OVERRIDES`. Visual hosts,
      Reddit links and Discord/Telegram/WhatsApp are rule-based.
- [ ] **Visual media = media that would have shown in the thread or post** — decided 2026-10-15 (user); these are not
      also counted as URL links (`aft_model/rules.py`, `link_renders`). Renders: link posts to Reddit media/galleries, the
      embed hosts, `media_host` domains (except the file lockers mega.nz, mixdrop, verystream) and direct image/video
      files on any host; in post text or comments only images uploaded into the text (preview.redd.it) and image/GIF
      markdown. Other media links in text showed as links, so they count as URL links. Pilot: 45,284 → 43,132 visual
      links; 2,345 text links move to URL links (61 accounts, one account 1,816 of them), 193 link posts move in
      (109 direct files on porn hosts, 48 media_host). Recheck counts on the larger sample; the rule assumes Reddit's own
      site, not third-party apps that previewed links in comments.

## Sample and data rules
- [x] **Test-subreddit accounts**: 7 pilot accounts used test subreddits; the 2 `lss_mobile` ones were Reddit internal test
      accounts → pool rule 8. The other 5 (incidental use) were kept.
- [ ] **`example_viral`** is in the pool and does not match rule 8 — check whether it is also an internal test account before
      the larger draw.
- [ ] **Coverage-limited variables** (removal, score, reply counts) under the collected-before-accusation rule: pilot had
      80/1,000 accounts with a qualifying post; recheck.
- [ ] **#8 reply collection** (`collect_replies_v4.py` → `replies_output/`): pilot = 27,795 post threads + 58,596 comments
      (~0.74 s and ~1 s per item at 4 workers; only 31% of comments get any reply). The 5 accounts at the 10,000-comment
      archive cap (46% of pre-accusation comments) were **skipped entirely** (user, 2026-10-08) and are dropped from the
      model (user, 2026-10-15; see Truncated histories). Budget collection time for the larger sample (and whether to skip capped accounts there too).
- [ ] **#11 karma per post/comment** uses final (archive-time) scores (user, 2026-10-08) — includes votes cast after the
      accusation; state this caveat, and compare with the subset archived before accusation (pilot: 69 accounts posts,
      205 comments) if the larger sample gives enough.
      Measured as four last-7-day counts: downvoted posts (score 0), downvoted comments (< 0), popular posts and comments
      (≥ 100). Pilot accounts: 360 / 140 / 463 / 106. **Recheck the 100 cut-off** and the counts on the larger sample.
- [ ] **Main vs extended model** (2026-10-08): main = dated behaviour + karma counts; extended = main + #19 removal flags
      (moderator/AutoModerator post removals; Reddit-level removals excluded as mostly suspension sweeps). Compare how
      much the main time ratios move. Checks: karma and removal on posts > 30 days before accusation vs last 30 days
      (leak check); removal on active accounts only; removal dated only (archived before accusation or a dated removal
      notice in the thread).
- [ ] **Hyak datasets** (UW CDSC cluster, `/gscratch/comdata/output/reddit_{submissions,comments}_by_{author,subreddit}.parquet`,
      Pushshift + Arctic Shift to March 2026; user's account pending 2026-10-08): use for full histories past the 10,000
      cap (pilot's 5 capped accounts; pool rule 2 drops), #18 copying beyond the thread, and thread downloads at scale.
- [ ] **Copied from anywhere on Reddit (full archive search, user 2026-11-17)** — to build once the full search lands.
      Rules (user): a text counts as copied only if the matching archive text was posted **before** the time the accused
      account posted it; only texts before the account's own accusation; nothing on or after 2022-11-30. Model terms
      (agreed 2026-11-17): per 7-day window, "posted a checkable text" (140+ characters), log(1 + checkable texts),
      log(1 + copied texts). Sensitivity: 50-character cut, dropping widely circulated copypasta, eligible accounts only.
      **Keep per text** the number of earlier matches and of distinct earlier authors (user: record it, not in the model
      unless she asks). CDSC rule: no comment text or usernames from the archive in any write-up.
- [ ] **#18 copied from other accounts**: pilot uses only (a) a comment copying the text it replied to and (b) copies
      inside the account's own post threads. Larger sample (Hyak): add copies anywhere in threads they commented in and
      post titles copied from anywhere on Reddit; compare with the pilot version to measure the undercount.
- [x] **Account age**: dropped from the model (2026-10-07), so undatable accounts (pilot: 185, all suspended) no longer matter.
- [ ] Pool rules (48-hour edit window, first action before first mention, dates) stay fixed — record any change.
