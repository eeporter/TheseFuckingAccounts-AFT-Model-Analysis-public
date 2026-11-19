# fastscan2: a faster version of the exact copy search

Separate copy. Nothing in `copy_search_fast/` or `lsh_copy_search/fastscan/` is changed.
CDSC rule: nothing here contains comment text or usernames. Benchmarks so far use made-up text only.

## What is the same as fastscan

Same cleanup rules, same word 3-grams, same xxhash64 (seed 42) shingle hashes, same exact prefix-filter index,
same exact Jaccard >= 0.7 check. Results are identical by construction, and checked (see below).

## What is different (speed only)

1. **One-pass cleanup in C.** fastscan cleans text with four pyarrow passes (lower case, three regex replacements),
   each copying the column. fastscan2 does lower case, URL removal, punctuation and whitespace in one C pass over the
   raw Arrow buffer. The per-character tables are computed from pyarrow itself at start-up, so Unicode behaviour is
   whatever pyarrow does in that environment. `test_clean_v1.py` checks the output is byte-identical.
2. **Length rule first, no copies.** The 140-character rule is applied in C on the raw buffer.
3. **Sort and verify only when needed.** A row's shingles are checked against a small in-cache bit filter; the row is
   only sorted, de-duplicated and verified if one of its shingles is actually indexed.
4. **Index ordered by an archive sample** (`Index(..., sample_shingles=...)`). fastscan indexes each query under the
   shingles that are rarest *among the queries*; many of those are everyday phrases that are common in the archive.
   fastscan2 can instead prefer shingles that are rare in a sample of archive rows. The prefix filter is exact under
   any fixed order, so this cannot change results; it only cuts the number of rows that need verifying.
5. Archive ids are read only for row groups that have matches.

## Rules version (added 17 Nov, afternoon)

`rules2.py` holds the current rules. The rule code (game-notice, markdown-link and URL patterns, the length + link
rule, the time conversion) is copied verbatim from fastscan's `fast_full_v1.py` (md5 f2d6e34e...), so both versions use
one definition. `fast2_full_v1.py` is the full-run script: same command line and same output columns as
`fast_full_v1.py`. Every earlier match is kept as its own row (with the earlier item's author, subreddit and time), so
the number of earlier copies and of distinct earlier authors per text can be counted afterwards.

How the rules are applied with less work, without changing the result:
- date mask first, as in fastscan;
- the game-notice pattern can only match text that contains "old reddit"; those rows (and invalid UTF-8) go through
  fastscan's exact pipeline. For all other rows the notice-stripped text is the raw text, so the C scanner is used;
- rows whose raw length is under the minimum cannot pass the length rule and are skipped;
- the length + link rule is evaluated only for rows that produced a match.

## Time conversion (18 Nov 2026)

The archive's `CreatedAt` is America/Los_Angeles local time WITH daylight saving (UTC-7 in daylight-saving months,
UTC-8 otherwise). `rules2.py` / `rules3.py` added a flat +8 h; `rules4.py` converts it properly (`archive_utc`). Every output
before `full_out_v4/` therefore has `a_created_utc` 1 h late in daylight-saving months and misses copies posted 0-1 h
before the query in those months. See the Results section of `../README.md`.

## Files

| File | What |
| --- | --- |
| fastscan2.c / fastscan2.py | the scanner |
| test_clean_v1.py | cleanup equality check: 1,412,064 generated strings (every Unicode code point + 300,000 random) |
| make_synth_v1.py / bench_v1.py | made-up benchmark data and the fastscan vs fastscan2 timing + equality check |
| bench_synth_2M_v1.json | result of that benchmark (below) |
| pilot_fast2_v1.py / run_pilot_fast2_v1.sh | Hyak check on the Spark pilot's comment file, pilot rules |
| rules2.py | the current rules (copied from fastscan) + fast and reference pipelines for one row group |
| fast2_full_v1.py | full-run script with all rules (same CLI and output as fastscan's fast_full_v1.py) |
| run_full_fast2_v1.sh | Slurm array script for the full run (private build per task, built-in check against fastscan's rule code) |
| rules_check_v1.py / run_rules_check_v1.sh | rules version vs fastscan's rule pipeline on the same rows: equality + timing |
| make_synth_rules_v1.py | made-up data exercising the rules |
| ref/ | unmodified copies of fastscan.c and fastscan.py, used only as the reference in checks |
| fast2_full_v2.py / run_full_fast2_v2.sh | full run with 12 workers per 4-core task (A/B test 43332090, full run 43332591) |
| rules3.py / fast2_full_v3.py / run_fix_v3.sh / run_merge_v3.sh | chunked pair writes (Arrow 2 GB limit) and fail-fast pool; rerun of tasks 1, 3, 22 (43360110) and merge (43360111) |
| rules4.py / fast2_full_v4.py / run_full_v4.sh | current version: daylight-saving aware archive time (`archive_utc`); whole-pool run 43362084 -> `full_out_v4/` |
| run_checks_v4.sh | checks on full_out_v4 (43362085): completeness, old vs new (`compare_dst_v1.py`), invariants v3, brute force v3 |
| compare_dst_v1.py | full_out_v3 vs full_out_v4: every old pair present, every new pair a daylight-saving item 0-1 h before the query |
| invariants_check_v3.py | per-row rules, no self-match, rows on disk = summaries (streaming) |
| bruteforce_check_v2.py / bruteforce_check_v3.py | all queries against a random archive slice, no prefix or Bloom filter (v2: +8 h, v3: rules4 time) |
| time_offset_check_v1.py / run_checks_v3c.sh | archive CreatedAt vs Reddit created_utc by month (43361281, step 1) |

## Run log

| Date | Where | What | Result |
| --- | --- | --- | --- |
| 2026-11-17 | cloud container, 1 core | `test_clean_v1.py` | PASS, 0 mismatches in 1,412,064 strings |
| 2026-11-17 | cloud container, 1 core | `bench_v1.py data_big orig 4`: 2,000,000 made-up archive rows (579,996 with 140+ characters), 15,866 made-up queries, 2,000 planted edited copies | 610 hits from each of fastscan, fastscan2, fastscan2 with sample order; hit lists identical. CPU: fastscan 13.8 s (about 140,000 rows/s), fastscan2 about 4.9 s, fastscan2 with sample order 3.6 to 3.9 s (about 510,000 to 540,000 rows/s, 3.5 to 3.8 times faster across two runs). Reading the file alone is 1.0 s. Rows needing sort + verify: 328,172 -> 58,618 with the sample order. |
| 2026-11-17 | Hyak job 43324646, 2 cores, 27 s | tiny check: `MAX_ROW_GROUPS=2 OUT_NAME=tiny_fast2_out_v1 sbatch -A comdata -p cpu-g2 -c 2 --mem 16G --time 0:15:00 -J tiny_fast2_v1 ... run_pilot_fast2_v1.sh`. 36,089 queries (full copypasta candidate set, 140+ characters) vs the first 2 of 27 row groups of the Spark pilot comment file (2,030,781 rows; 595,591 with 140+ characters) | **PASS.** Cleanup check 0 mismatches in 1,412,064 strings (pyarrow in the CDSC container). 32 pairs, the same 32 with the same counts as fastscan's saved pairs for those row groups; none found by only one. 2.8 worker-seconds, about 725,000 rows/s per core (fastscan's whole-file figure is 142,000). 49,190 rows needed sort + verify. Caveat: the sample used for ordering (row group 0, 1,017,331 rows) is half of what was scanned, and it is a 2-row-group slice, so the whole-file run is the fair timing. |
| 2026-11-17 | cloud container, 1 core | `rules_check_v1.py` on 1,000,000 made-up archive rows with game notices, links, dates around the cutoff; 15,866 made-up queries (11,169 after rules); at 140 and at 50 characters | **PASS** both. 122 pairs each, identical on every output column. About 2.9 times (140) and 3.3 times (50) faster than fastscan's rule pipeline. |
| 2026-11-17 | cloud container, 2 cores | fastscan's real `fast_full_v1.py --minchar 50` vs `fast2_full_v1.py --minchar 50` on the same made-up archive | Same schema, same 122 rows, Jaccard identical; 9.6 vs 3.2 CPU-seconds. |
| 2026-11-17 | Hyak job 43328586, 8 cores, 40 s | `sbatch -A comdata -p cpu-g2 -c 8 --mem 32G --time 0:30:00 -J pilot_fast2_v1 ... run_pilot_fast2_v1.sh`: whole Spark pilot comment file (26,922,660 rows; 7,786,927 with 140+ characters), 36,089 queries, pilot rules, same setup as fastscan's pilot job 43321861 | **PASS.** 427 pairs, the same 427 with the same counts as fastscan; none found by only one. Scan 8.1 s wall on 8 cores. Worker time summed: 56.4 s wall (35.0 s CPU) against fastscan's 189 s, about 3.4 times faster; about 477,000 rows/s per core against 142,000. Rows needing sort + verify: 1,216,737. Peak memory 4.7 GB. |
| 2026-11-17 | Hyak job 43328587, 2 cores, 83 s | `sbatch -A comdata -p cpu-g2 -c 2 --mem 16G --time 0:20:00 -J rules_check_v1 ... run_rules_check_v1.sh`: ALL RULES, first 2 of 18 row groups of older-batch comment file part-00000-074ac9d9 (2,088,508 rows; 1,264,687 before the date limit), query scope pre2022, fastscan's rule pipeline (ref/) and fastscan2 on the same rows in one process | **PASS at 140 and at 50.** 140: 27,224 queries (396 accounts), 6 pairs from each, identical on every column; scan CPU 12.2 s vs 2.58 s (4.7 times). 50: 71,420 queries (722 accounts), 7 pairs from each, identical; 13.35 s vs 2.84 s (4.7 times). Only 48 / 57 rows needed the game-notice slow path. One-off sample + index build: 6.5 to 7.6 s per task. |
| 2026-11-17 | cloud container, 3 cores | fastscan's real `fast_full_v1.py --minchar 50` vs `fast2_full_v1.py --minchar 50 --selfcheck --flush-rows 40` on made-up comments (1,000,000 rows) AND made-up posts (300,000 rows, title + selftext, some empty) | Same schema, same 144 rows (122 comment, 22 post), Jaccard identical; built-in check passed for both kinds; output split into 4 part files as intended. |

## FULL RUN (17 Nov 2026): Hyak array job 43329112, all 30 tasks COMPLETED

Command: `sbatch -A comdata -p cpu-g2 --array=0-29 -c 4 --mem 32G --time 2:00:00 -J full_fast2_v1 -o .../fastscan2/logs/full_fast2_v1_%A_%a.out run_full_fast2_v1.sh`
(`fast2_full_v1.py --minchar 50 --query pre2022 --files older --selfcheck`). Output: Hyak `reddit_copy_check/fastscan2/full_out_v1/`
(36 pair files, 95 MB, plus summary_taskNNNN.json). Count-only logs and summaries: Mac `ClaudeProjects/copy_search_fast2/logs/full_fast2_v1/`.

| | |
| --- | --- |
| Archive read | 2,000 of 2,000 older-batch files, 26,593 row groups, 23,604,630,484 rows (comments 20,703,221,154; posts 2,901,409,330) |
| Rows before the date limit | 15,958,060,585 (comments 14,066,612,247; posts 1,891,448,338) |
| Rows cleaned and shingled (50+ raw characters) | 9,348,406,152; 710,310 went through the game-notice slow path |
| Queries | 71,420 texts from 722 accounts (50+ characters, before 30 Nov 2022, rules applied) |
| Pairs (Jaccard >= 0.7, archive item earlier than the text, 50+ characters) | 24,277,320 (comments 21,520,549; posts 2,756,771). Task 27 alone has 9,564,193. Not yet tallied per text or filtered to 140. |
| Built-in check | 60 of 60 row groups ok (30 comment + 30 post): 4,804 pairs from fastscan's rule code, the same 4,804 from fastscan2 |
| Timing | submitted 15:45:47 PST, first start 15:47:47, last end 16:33:44: 47 min 57 s. 29 tasks ran together, 23 min 41 s to 27 min 08 s each; task 29 waited 29.5 min for cores (128-core limit) and ran 16 min 26 s. Without that wait: about 28 min. |
| Cost | 49.6 core-hours reserved; 10.35 CPU-hours used. CPUs were busy about 21% of the time: reading the parquet files from /gscratch was the limit, not matching. 128,000 to 217,000 rows/s per core (wall). Peak memory 7.2 GB. |

Lesson for a rerun: it is disk-bound. More cores per task will not help; fewer tasks reading at once, or simply accepting ~25 min, is fine.


## Handoff for the AFT model: what the full-run output is and how to use it

This section describes the first run (`full_out_v1`, 987 accounts). The current output is `full_out_v4` (whole pool,
daylight-saving aware time): see the Results section of `../README.md`.


- **Pairs** (one row per accused text x earlier archive item): Hyak `/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/full_out_v1/pairs_task*_p*.parquet`,
  24,277,320 rows. They contain usernames (`a_author`), so they stay on Hyak and are never printed, copied off Hyak or committed.
- **Columns:** `ckey` (accused text, `kind:id`, same key as `lsh/candidates_v1.parquet`), `aid`, `akind` (comment/post), `a_author`, `a_subreddit`,
  `a_created_utc`, `q_created_utc`, `q_acc_utc` (the account's accusation time), `q_vis_len`, `a_vis_len` (characters not counting links),
  `inter`, `a_nsh`, `q_nsh`, `jaccard`. All times are UTC seconds. In `full_out_v1` the archive time used a flat +8 h, so `a_created_utc` is
  1 h late in daylight-saving months; use `full_out_v4` (whole pool, daylight-saving aware) for anything new.
- **Rules already applied:** normalized word 3-grams, exact Jaccard >= 0.7; game notice stripped; links at most half of the text and not counted
  toward length; archive item strictly EARLIER than the accused text; nothing on or after 30 Nov 2022 UTC (2025-26 batch skipped);
  both texts 50+ characters.
- **140-character version:** keep rows with `q_vis_len >= 140` and `a_vis_len >= 140`. Checked identical to a separate 140 run on the test slice.
- **100-character version (Emily asked for it 17 Nov 2026):** the same filter with 100. No new search is needed, because the run kept everything from 50 up.
- **Pre-accusation only:** keep texts with `q_created_utc < q_acc_utc`.
- **Every earlier match is kept**, so one text can have thousands of rows (viral copypasta, bots, the account's own earlier posts). A count of
  earlier copies is therefore not the same as "copied from someone else": use the distinct-other-author columns of the tally below.
- **Searched texts (the denominator):** 71,420 texts from 722 accounts at 50 characters; 27,224 texts from 396 accounts at 140.
  Accounts with no text before the cutoff were never searched (not the same as "searched, no copy").

### Tally job (written 17 Nov 2026; run as job 43331456 on `full_out_v1`, before the daylight-saving fix)

`tally_fast2_v1.py` / `run_tally_fast2_v1.sh`. Reads `full_out_v1/` and `lsh/candidates_v1.parquet`; writes Hyak `fastscan2/tally_out_v1/`:

| File | Rows | Columns |
| --- | --- | --- |
| `searched_texts_v1.parquet` | every searched accused text (expect 71,420), matches or not | `ckey`, `kind`, `outer_key` (the ACCUSED account's key, as in candidates_v1), `q_created_utc`, `q_acc_utc`, `q_vis_len`, `searched_100`, `searched_140`, `pre_accusation` |
| `per_text_tally_v1.parquet` | every text with at least one earlier copy at 50 | `ckey`, `q_created_utc`, `q_acc_utc`, `q_vis_len`, then for S = 50, 100 and 140: `n_copies_S`, `n_authors_S`, `n_self_copies_S` (same account), `n_deleted_copies_S` (author `[deleted]`), `n_other_copies_S`, `n_other_authors_S`, `n_subreddits_S`, `first_seen_utc_S` |
| `tally_summary_v1.json` | counts only | totals and two sanity checks that must be 0 |

No comment text and no matched-author usernames are in these files. `outer_key` identifies the accused account, so the files are for analysis
on Hyak/bia only and must not be published. After the job, the two parquet files go to bia
`~/Dissertation/Reddit_project2_notebooks/p2_analysis/copy_search_output/` with a copy of this section.

Submit: `sbatch -A comdata -p cpu-g2 -c 4 --mem 48G --time 0:30:00 -J tally_fast2_v1 -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/tally_fast2_v1_%j.out run_tally_fast2_v1.sh`
