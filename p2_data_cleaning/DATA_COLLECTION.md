# Project 2 — data collection record

Collected 2026-09-28 → 2026-10-05. Everything below is reproducible from
`collection_scripts/`; every script is resumable and checkpointed.

**Final analysis table: `data/aft_analysis_table.parquet` — 55,520 accounts × 60 columns.**

---

## 1. Population: who counts as an accused account

**Source: submissions only.** Subreddit comments (84,467 of them) were deliberately *not*
used — the accused/victim disambiguation rules below are validated against the submission
house style and do not hold in conversational replies.

| Corpus | Posts | Range |
|---|---|---|
| `/data/Emily_data/TheseFuckingAccounts_submissions.csv` | 17,631 | 2014-12-21 → 2022-12-31 |
| Arctic Shift `posts/search?subreddit=TheseFuckingAccounts` | 4,245 | 2023-01-01 → 2026-09-29 |
| Combined, after dedupe on `(title, selftext, url)` | **21,469** | 2014-12-21 → 2026-09-26 |

The CSV ends at 2022; the subreddit is still active, so the archive supplies 3.75 further
years. Dedupe includes `url` because two link posts can share a title while naming
different accounts — the notebook's original `(title, selftext)` key collapsed them.

### Extraction rules (12)

Base rules are cells 8–12 of `collect_accused_usernames_and_first_mention.ipynb`. Seven were added
after inspecting posts where accounts were visibly missed.

| Method | Accounts | Catches |
|---|---|---|
| `explicit_mention` | 72,008 | `u/name`, `reddit.com/user/name` |
| `title_lead_rare` | 1,065 | `ExampleNameA: 2 hours, low-effort posts` |
| `submission_url` | 809 | link posts whose title never names the account |
| `title_overview` | 580 | `overview for examplenameb` |
| `title_lead_corroborated` | 119 | bare lead name seen with `u/` elsewhere |
| `title_age` | 69 | `example-101 - 7 days old` |
| `title_lead_comma` | 67 | `ExampleNameC, another copy/paste bot` |
| `title_username_list` | 45 | `examplename1 examplename2 examplename3` |
| `title_lead_run` | 25 | `exname56, exname5456, exname5264. x-posting…` |
| `title_born` | 14 | `ExampleTheName - born on January 3, 2017` |
| `title_possessive` | 9 | `Update- ExampleSurname's 8 months old account` |
| `title_woke` | 2 | `Examplename18, woke up today after 5 years` |

**`submission_url` was the largest gap.** The extractor only ever read `title` and
`selftext`, never the link the post points at — yet for a link post the profile URL *is*
the accusation. 2,952 posts link to a profile; in 1,869 the URL names an account the title
does not. It also beats the title where the reporter mistyped (`examplenamee-` vs `examplename`).

**Corroboration by rarity.** The notebook accepted a bare leading name only if it appeared
with a `u/` prefix somewhere in the corpus, which missed accounts nobody ever prefixed. The
replacement signal is derived from the same corpus: a token appearing **fewer than 3 times
in ordinary prose** is distinctive. Separation is sharp — `Request` (95), `Discussion` (92),
`AMA` (36), `Tip` (30), `FYI` (9) above it; `qwzxkjv`, `exampip`, `Exampletri` at 1.

**Shape test** (`collection_scripts/username_shape.py`) gates every bare-name rule: a username has a
digit, underscore, hyphen or internal capital, is not all-digits, is not `r_`-prefixed
(subreddit shorthand), and — for hyphen-only tokens with no digits — is not an English
compound whose every part is common prose. The hyphen restriction is deliberate:
`copy-paste` and `karma-farming` are rejected while `well_that_example_went` and
`Mr_Example__Wild_Ride` survive, because usernames compound with underscores.

### Role exclusion (`collection_scripts/exclude_helpers_and_victims.py`)

Applied **per post**, so a name excluded in one post stays if accused in another.
319 mentions excluded, removing 39 accounts outright.

- **Victims** — the copied-from account: the notebook's possessive (`u/X's submission`) and
  attribution (`from / stolen from / originally by`) rules, plus the bare form
  `copy/paste of u/X submission` with no possessive.
- **Helpers** — credited with finding the account: `detected/reported/found/spotted/caught
  by u/X`, `thanks to`, `courtesy of`, `h/t`, `shoutout to`, `as u/X reported`, and
  `u/X pointed out / noticed / helped`.

Three traps encoded after they produced wrong removals:
1. **`OP is u/X` is NOT a victim marker here** — `OP is /u/example_op with comments from…`
   names the author of the *suspicious* post. No rule for it.
2. **Helper verbs must be on the same line** — `u/example_reporter12\n\nReported yesterday` is an
   accused account followed by the reporter's own sentence.
3. **Passive vs active** — `u/example_accused_a spotted in r/x` means the account *was* spotted
   (accused); `u/example_helper spotted them` is a helper. Ambiguous verbs require an object.

Notable removal: **`example_janitor_bot`**, an anti-spam bot named in the footer of 290 bulk
reports (*"These potential bots were detected by u/example_janitor_bot"*). It would otherwise
have been the most-accused account in the corpus at 238 mentions.

### Funnel

| Stage | Accounts |
|---|---|
| Extracted from the combined corpus | 74,812 |
| − deleted (Reddit 404) | −12,881 |
| − no Arctic Shift presence (nothing to collect) | −6,411 |
| **Final analysis table** | **55,520** (37,969 suspended / 17,551 active) |

---

## 2. Collection tiers

| Tier | Source | What | Coverage |
|---|---|---|---|
| 2 | PRAW (`oauth.reddit.com`) | status + 26-field profile | 61,931 / 61,931 |
| 3 | Arctic Shift `users/search` | `_meta` activity span, counts, karma | 61,931 / 61,931, 0 errors |
| 4 | Arctic Shift `posts`/`comments/search` | full item history | 55,520 / 55,520 eligible |

**Tier 2** is the only source of a true account creation date (`created_utc`), and only for
active accounts — a suspended account returns 6 fields and no dates. Also captures
`has_verified_email`, `has_subscribed`, `accept_pms`, `icon_img` (default avatar), karma.

**Tier 3** works for suspended accounts, which is why it exists. Note `_meta` counts are
archive-wide rollups refreshed at fixed dates (posts 2025-08-26, comments 2025-03-25), so
they are a **lower bound** and their `last_*_at` fields are capped at those dates. Take
last-activity from the *history*, never from `_meta`.

**Tier 4** is capped at **10,000 items per stream**; 441 accounts hit it and carry
`history_truncated = True` with true totals in `expected_posts` / `expected_comments`.
Uncapped it would have been 39.4 h and tail-dominated — 335 accounts held 42.5 M of the
59.6 M items, one with 4.4 M alone. Accounts are processed cheapest-first so partial
completion covers the most accounts.

---

## 3. Bugs found and fixed

1. **Cursor skipped tied timestamps.** Arctic Shift treats `after` as **exclusive**, so
   `after = data[-1]['created_utc']` silently drops every remaining item sharing that
   boundary timestamp. Verified: `u/example-acct` returned 11,800 of 11,857 comments, with 38 of
   118 page boundaries on a tied timestamp. **Fix: `after = max(created_utc) - 1` + dedupe
   by `id`.** Re-pulling all 7,895 paginated accounts recovered **5,429 posts and 3,331
   comments** across 434 accounts. The loss is undetectable after the fact — the skipped
   items are the evidence — so only a re-pull measures it.
2. **Part files had incompatible column types**, so `pd.read_parquet('<dir>')` raised
   `ArrowNotImplementedError: Unsupported cast from string to null`. A batch where a column
   was all-`None` got typed `null`; int64/double drifted with missing values. Reading a
   column subset hides it. **Fix: one unified schema across files, cast each on read.**
3. **43% duplicate rows** — the repair appended re-pulled rows beside the originals (by
   design, so a short re-pull could not destroy data: 216 of 8,227 returned fewer).
   **Fix: deduplicate into `data/history_*`.**
4. **2.1 M rows for accounts later found deleted or role-excluded** — Tier 4 began before
   Tier 2 finished. **Fix: filtered on consolidation.**
5. **Short-page stop condition.** `if len(data) < 100: break` would truncate on a transient
   short page. Replaced with "stop when a page yields no new ids". Measured on 20 accounts:
   recovers **0 items** — the API does not emit spurious short pages — so this is insurance
   for future runs, not a repair. The collection was **not** re-run for it.
6. **8 Tier 3 lookups** had unretried transient timeouts. Retried, 8/8 resolved.

### Operational lessons

- **Throughput is latency-bound, not limiter-bound.** Serial requests run at ~1.0/s because
  each round trip takes ~1 s, so `LimiterSession(per_second=5)` never engages. 5 worker
  threads sharing **one** limiter give 4.8/s with no 429s; 8 workers add nothing.
- **`users/search` takes one author per request.** `author=a,b` returns HTTP 200 and zero
  rows — a silent wrong answer. Assert one row back per name.
- **`/api/<kind>/ids?ids=` truncates above 100 ids**, returning 200. Batch at 100 and assert.
- **`meta=true` is rejected** (400). The `arctic-shift-pull` skill is stale on this.
- PRAW sustains ~1.7/s; Arctic Shift and Reddit are separate quotas and can run in parallel,
  but never run two processes against the *same* service.

---

## 4. Files

| Path | Contents |
|---|---|
| `data/aft_analysis_table.parquet` | **55,520 × 60 — the modelling table** |
| `data/accused_accounts_final.parquet` | 74,812 accused accounts, all statuses |
| `data/accused_accounts_not_deleted.parquet` | 61,931 after dropping deleted |
| `data/history_posts/` | 12,149,311 posts, deduped & filtered (1.01 GB) |
| `data/history_comments/` | 10,947,587 comments, deduped & filtered (1.06 GB) |
| `data/history_summary.parquet` | per-account counts, first/last activity, subreddit counts |
| `data/account_status_checkpoint_full.parquet` | status for 74,865 accounts |
| `data/account_profiles_full.parquet` | 26-field PRAW profiles |
| `data/account_meta_arctic.parquet` | Tier 3 `_meta` |
| `data/arctic_pull_done.parquet` | Tier 4 ledger (counts PRE-repair; use `history_summary`) |
| `data/cursor_repair_ledger.parquet` | what the cursor repair recovered, per account |
| `data/excluded_role_mentions.parquet` | the 319 helper/victim exclusions, auditable |
| `data/tfa_submissions_2023_2026.parquet` | the 4,245 newer submissions |
| `data/arctic_posts/`, `data/arctic_comments/` | raw pull, pre-dedupe — audit trail only |
| `data/extraction_sample_100_v3.csv` | 100-post extraction sample for manual review |

`data/` is gitignored. `collection_scripts/` and this file are tracked.

**Read the history with the canonical dirs** (`data/history_posts`), not the raw ones.

---

## 5. Notes for the AFT model

- **Suspension date is never observable.** Reddit returns `is_suspended` with no date. The
  event time is **interval-censored**: suspension lies in `(last_activity_at, status_checked_utc]`.
  Fit `lifelines` `WeibullAFTFitter.fit_interval_censoring`; midpoint imputation is a
  sensitivity check, not the primary model — intervals can be a decade wide.
- **Origin:** `praw_created_utc` is the true birth date but exists only for the 17,551
  active accounts. Otherwise `first_activity_at`. The gap between them is measurable on
  active accounts (78 days on the one checked) and characterises the proxy's bias.
- **Left truncation:** an account enters the sample only by surviving long enough to be
  accused. Delayed entry at `first_mention_date` is the cleaner specification.
- **Detection is on the causal path**, not a confounder: being posted about plausibly
  *causes* suspension via admin reports. `times_mentioned` conditioning changes the question.
- **Deleted accounts were dropped** (12,881); some are suspended-then-deleted or evasion, so
  this is informative censoring that biases suspension downward.
- **"Active" mixes two states** — currently posting vs quietly abandoned. Consider competing
  risks, or at minimum a days-since-last-activity covariate.
- **Status is not static:** re-checking the original seed moved 15 accounts active → deleted
  over seven weeks. `status_checked_utc` is recorded per row.
