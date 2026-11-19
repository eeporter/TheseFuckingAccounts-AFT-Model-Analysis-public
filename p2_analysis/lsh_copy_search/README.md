# Exact copy search (copypasta) for accused accounts

This folder holds the code that finds earlier near-copies of accused accounts' Reddit texts in the CDSC Reddit
archive on Hyak. CDSC rule: nothing here contains comment text or usernames; logs and summaries are counts and ids only.
Data (parquet) and pair outputs are not in git.

Contents in historical order:

| Path | What |
| --- | --- |
| `PLAN.md`, `lsh_*_v1.py`, `make_toy_v1.py`, `build_candidates_v1.py`, `run_*_v1.sh` | Spark MinHash LSH version (12 Nov 2026), replaced by the exact search below |
| `fastscan/` | First exact scanner (C + Python): prefix-filter index, exact Jaccard. Toy and pilot runs |
| `fastscan2/` | Current exact scanner: same results, much less work per archive row (Bloom filter, one-pass cleanup in C). All full runs and checks; current version `rules4.py` + `fast2_full_v4.py` (daylight-saving time fix) |
| `fastscan2/ref/` | Copy of `fastscan/fastscan.{c,py}`, used only by `--selfcheck` |
| `build_candidates_v2.py` | Builds the whole-pool query set (38,106 accounts) |
| `../copy_search_hyak_v1.py`, `../copy_search_combine_hyak_v1.py`, `../copy_search_hyak_v1.sbatch` | Older substring-match copy search (Aho-Corasick, job 43145502), kept for the record |

Every `.py`, `.c` and `.sh` file has a header saying what it does, its inputs and outputs and which jobs used it.

## Method

Exact set-similarity join between a set of query texts (the accused accounts' comments and posts) and the archive.

- **Cleanup** (same as the earlier Spark job): lower case, `http(s)://` URLs removed, non-word characters to spaces,
  whitespace collapsed, trimmed.
- **Shingles**: word 3-grams `"w1 w2 w3"`, hashed with xxhash64 (seed 42, as in Spark). Each text is the *set* of its
  shingle hashes.
- **Duplicate** = exact Jaccard >= 0.7: `c / (|A| + |B| - c) >= 0.7` with `c` the number of shared shingles, tested in
  integers as `10c >= 7(|A| + |B| - c)`.
- **Prefix filter** (Chaudhuri, Ganti & Kaushik, *A Primitive Operator for Similarity Joins in Data Cleaning*, ICDE 2006,
  doi:10.1109/ICDE.2006.9): a query with `n` shingles indexes only its first `n - ceil(0.7 n) + 1` shingles in one fixed
  global order. Any text with Jaccard >= 0.7 shares at least `ceil(0.7 n)` of the query's shingles, so it must contain one
  indexed shingle. Exact under any fixed order; the order only changes speed. fastscan ranks shingles by how rare they
  are among the queries; fastscan2 ranks them by how rare they are in an archive sample (row group 0 of the task's first
  file), then among the queries.
- **Length filter** (Bayardo, Ma & Srikant, *Scaling Up All Pairs Similarity Search*, WWW 2007,
  doi:10.1145/1242572.1242591): only pairs with `0.7|A| <= |B| <= |A|/0.7` are verified. Prefix and positional filtering
  for near-duplicate detection are developed further in Xiao, Wang, Lin & Yu, *Efficient Similarity Joins for Near
  Duplicate Detection*, WWW 2008 (doi:10.1145/1367497.1367516); the positional and suffix filters from that paper are
  not used here.
- **Inverted index**: hash table from each indexed shingle to the list of queries that indexed it (postings). Each
  archive row looks up all of its shingles; queries found become candidates and are verified by exact sorted-set
  intersection.
- **Bloom filter** (fastscan2 only): a bit array with >= 16 bits per indexed shingle and one bit per shingle (about 6%
  of bits set), in front of the hash table. Bit 0 = definitely not indexed; bit 1 = look up the table. No false
  negatives, so results are unchanged; it lets most archive rows be rejected from CPU cache.
- **Runs**: Slurm array, task `i` of `N` takes every `N`-th archive file; inside a task the row groups are spread over
  worker processes that share one index (fork).

## Rules

- Query side (`--query pre2022`): texts created before 30 Nov 2022. Archive side: the older file batch only (through 2024;
  1,200 comment + 800 submission files); rows on or after 30 Nov 2022 are dropped before any text work.
- Archive `CreatedAt` is America/Los_Angeles local time WITH daylight saving (UTC-7 in daylight-saving months, UTC-8 otherwise), converted to UTC by `rules4.archive_utc` (measured 18 Nov 2026, job 43361281, step 1). Outputs before `full_out_v4` (`full_out_v1`, `full_out_v2`, `full_out_v3` and anything built from them) used a flat +8 h, so their `a_created_utc` is 1 h late in daylight-saving months and they miss copies posted 0-1 h before the query in those months.
- **Earlier-only**: a pair counts only if the archive item is strictly earlier than the query text. This also removes
  self-matches.
- Length and link rule (both sides, `--minchar 50`): the Reddit game-post notice is removed; for the length test,
  markdown links are reduced to their visible words and bare URLs removed, and at least 50 visible characters must
  remain; links may be at most half of the characters. Shingles are taken from the text with only the game notice
  removed (cleanup then drops `http(s)://` URLs). The output keeps both texts' visible length, so the 140-character set
  is a filter on the 50-character output.
- Post text = title + newline + selftext.

## Query sets

| Set | Accounts | Texts | Query texts (pre2022, 50+ chars) | Built by |
| --- | --- | --- | --- | --- |
| v1 (`candidates_v1.parquet`) | 987 (pilot sample after rule B) | 194,951 | 71,420 (722 accounts) | `build_candidates_v1.py` |
| v2 (`candidates_v2.parquet`) | 38,106 (eligible pool of 38,334 after rule B, no 1-year window) | 8,771,455 | 3,318,585 (28,068 accounts) | `build_candidates_v2.py` |

## Runs (Hyak, account comdata)

| Job | What | Result |
| --- | --- | --- |
| 43145502 | Older substring search (`copy_search_hyak_v1.sbatch`), 140 tasks | Superseded |
| 43301873 / 43301955 | fastscan toy test | Fail (container bind) / PASS: 33 pairs, same as brute force and Spark |
| 43321861 | fastscan, Spark pilot comment file | PASS: all 427 Spark matches, no extras |
| 43324646 / 43328586 | fastscan2 tiny check / pilot file | PASS: identical to fastscan |
| 43328587 | fastscan2 rules check | PASS |
| 43329112 | **Full run 1**, v1 queries, 30 tasks x 4 cores | 2,000 files, 23,604,630,484 rows, 24,277,320 pairs |
| 43331456 / 43331457 | Tally / sample page of full run 1 | counts only |
| 43332089 | Disk read-speed probe (4/12/24/48 readers) | 163 / 401 / 681 / 1,187 MB/s |
| 43332090 | 4 vs 12 workers A/B | identical pairs; 12 workers faster |
| 43332591 | **Full run 2**, v1 queries, 12 workers | about 25 min end to end |
| 43332603 | Full run 1 vs 2 | IDENTICAL, 24,277,320 pairs |
| 43340006 | **Whole-pool run**, v2 queries, 30 tasks x 4 cores x 48G | 27 of 30 tasks done in 44 min; tasks 1, 3, 22 failed (Arrow 2 GB string limit / out of memory) |
| 43340007 | Checks on the incomplete output | brute force IDENTICAL (186,762 pairs); invariants out of memory |
| 43360110 | Rerun of tasks 1, 3, 22 with `fast2_full_v3.py` (chunked writes) | all DONE in 18-20 min |
| 43360111 | Merge of the rerun into `full_out_v3/` | 30 of 30 summaries |
| 43360112 | Final checks on full_out_v3 (`run_checks_v3b.sh`) | complete; regression IDENTICAL (24,277,320); invariants: 0 rule violations, 121 nulls, 7,635 duplicate pairs; brute force IDENTICAL (186,762) |
| 43361281 | Follow-up checks (`run_checks_v3c.sh`) | step 1 time offset: daylight saving present (see Results); step 2 hit the 12 h limit, step 3 did not run (both covered by 43362085) |
| 43362084 | **Whole-pool run with the daylight-saving fix** (`run_full_v4.sh`, `fast2_full_v4.py`, `rules4.py`) | 30 of 30 tasks DONE, about 1 h of clock time; output `full_out_v4/` (current) |
| 43362085 | Checks on full_out_v4 (`run_checks_v4.sh`) | see Results |

## Results (counts only)

**Current output: `full_out_v4/` on Hyak** (whole-pool search, array 43362084, `rules4.py` / `fast2_full_v4.py`, 18 Nov 2026):
3,318,585 query texts from 28,068 of the 38,106 accounts at minchar 50; 2,000 archive files, 23,604,630,484 rows;
6,283,863,008 pairs; 27.2 CPU-hours, about 1 h of clock time including queuing; self-checks 60 of 60 ok. A clean run is
limited by shared-disk read speed (about 3.5 GB/s for the whole run), not CPU.

**Superseded:** `full_out_v3/` (flat +8 h; array 43340006 + rerun 43360110 + merge 43360111), 6,268,981,564 pairs.

Checks:
- Regression against the 987-account run (checks_v3b 43360112): the 71,420 old queries give 24,277,320 = 24,277,320
  pairs, identical in all 14 columns.
- Time offset (checks_v3c 43361281, step 1): 227,893 archive rows that are also query texts, compared with Reddit's
  created_utc from Arctic Shift: offset 7 h for 139,101 and 8 h for 88,792, split exactly at the US clock changes.
- Old vs new (checks_v4 43362085, step 2): all 6,268,981,564 old pairs are present in `full_out_v4` (time corrected);
  the 14,881,444 new pairs are all daylight-saving archive items 0-1 h before the query (0 exceptions).
- Invariants (checks_v4, step 3): 0 violations of the Jaccard, earlier-only, cutoff, length and self-match rules; rows on
  disk = pairs in the summaries. 121 null values and 7,643 duplicate (query, archive item) pairs, cause not yet traced
  (121 / 7,635 in `full_out_v3` too): deduplicate on (`ckey`, `akind`, `aid`) before counting copies.
- Brute force (checks_v4, step 4): all 3,318,585 queries against 51,998 random archive rows: 64,350 = 64,350 pairs,
  identical. Earlier brute force on `full_out_v3` (checks_v3b): 186,762 = 186,762, identical.

Caveats: the brute-force checks share the rule code (`clean_and_keep`) and the reference cleanup (`ref/fastscan.norm`)
with the search, so they are not fully independent of it. The old-vs-new comparison uses sums of 64-bit row hashes, so it
is exact up to 64-bit hash collisions.

## Output columns (pair files)

`ckey` (query kind:id), `aid`, `akind`, `a_author`, `a_subreddit`, `a_created_utc` (UTC seconds), `q_created_utc`,
`q_acc_utc` (query account's accusation time), `q_vis_len`, `a_vis_len`, `inter` (shared shingles), `a_nsh`, `q_nsh`,
`jaccard`. One row per (query text, earlier archive item).
