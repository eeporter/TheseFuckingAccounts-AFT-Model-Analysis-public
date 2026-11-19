# Near-duplicate (copypasta) search with Spark MinHash LSH on Hyak

Plan agreed with Emily on 2026-11-12. Code lives here; data stays on bia/Hyak (parquet is gitignored).
CDSC rule: never publish comment text or usernames. Outputs carry ids, metadata and scores only.

## Names
- **Full copypasta candidate set**: all comments and posts of the 987 accused accounts in the model sample
  (sample_998_v4_excl_output, minus the capped accounts): 92,067 comments + 102,884 posts = 194,951 texts.
  Built by `build_candidates_v1.py`. Post text is title + selftext.
- **Toy set**: the planted test inputs (1,190 texts), built by `make_toy_v1.py`. Never mixed with the above.
- **Archive**: CDSC Reddit parquet on Hyak, through March 2026.
  Comments 25.9B rows / 3.2 TB (1,400 files); submissions 3.57B rows / 789 GB (1,000 files).

## Environment (checked 2026-11-12)
- Spark 4.2.0 (Scala 2.13, Java 25, Python 3.13) in `/gscratch/comdata/containers/cdsc_base/cdsc_base.sif`;
  `pyspark.ml.feature.MinHashLSH` available. The container only starts on compute nodes.
- CDSC default is single-node `local[*]`. Multi-node standalone scripts exist in `/gscratch/comdata/local/spark_config`
  but their README says the cross-node path has never been exercised since the July 2026 container move.
  A two-node test is a separate, later question (Emily wants to know whether it works).
- comdata allocation: cpu-g2 128 cores / 1 TB, cpu-g2-mem2x 32 / 500 GB; ckpt-g2 (~120 nodes, preemptible) for scale.
- Shared scratch `/gscratch/scrubbed` was 96% full (6.2 TB free), so Spark local dirs go to node-local /tmp (1.2 TB NVMe).

## Method
- Normalise: lower case, URLs removed, punctuation to spaces, whitespace collapsed.
- Shingles: word 3-grams (character 5-grams are a possible alternative; can be compared on the toy set).
- Signatures: HashingTF (2^28, binary) then Spark `MinHashLSH` with 100 hashes; our own AND-banding,
  20 bands x 5 hashes (Spark's own approxSimilarityJoin ORs single hashes, which floods buckets with common phrases).
- Join: the candidate side's band keys are small and broadcast; the archive side streams through and is never
  shuffled. Pass 2 re-reads only matched rows and computes exact Jaccard on shingle sets.
- **Dupe = exact Jaccard >= 0.7** (we expect exact or near-exact copypasta). 20x5 bands find ~97% of pairs at 0.7.
- **Minimum length: ignore texts under 140 characters** (pilot and full run, on both sides). This replaces the
  short-text exact-hash path used in the toy run.
- Scale-out: split the archive files into slices, one single-node Spark job per slice (Slurm array, restartable),
  rather than one multi-node Spark cluster. The data is read from GPFS (/gscratch) by every node.
- Rough cost: ~29.5B rows; 400-1,600 core-hours; 15-45 min wall clock on ~20 ckpt nodes. To be measured.

## Outputs
- Per-text summary: id, kind, dupes (J >= 0.7), near-exact (J >= 0.9), distinct other authors, subreddits, first/last seen.
- Pair table (kept for the toy; keeping it for the full run is undecided): id, match id, Jaccard, bands hit,
  same author / self, match subreddit and time.

## Steps
1. Toy planted test (`make_toy_v1.py`, `lsh_toy_v1.py`, `run_toy_v1.sh`): toy set vs one comment file (~18M rows).
   Plants from 100 real comments: reformatted, 5-50% words swapped, truncated, text appended; short texts; random-word
   negatives. Checks the code path, recovery vs the expected banding curve, output format and timing.
2. Pilot: full copypasta candidate set vs ~1% of archive files; measure throughput, bucket skew, matches per text.
3. Full run: array jobs over all comment and submission files.

## Decisions and findings after the pilot (2026-11-12)
- **Only earlier copies count:** a match counts only if the archive item was created before the accused text (archive CreatedAt < candidate created_utc).
- **Timezone:** archive CreatedAt is stored 8 hours behind UTC (written in Pacific time); convert to UTC before comparing with created_utc.
- **Game posts:** 3,719 posts in the full copypasta candidate set (3,718 from one account in r/SwordAndSupperGame) carry Reddit's fixed notice that the post is not supported on old Reddit, plus a game-generated title. Strip that notice on both sides before the 140-character cut.
- Archive files come in two batches: an older one (through 2024) and a newer one (2025 to March 2026). Pilot files must come from the older batch and contain sample accounts, so earlier matches and self-matches can occur.
- Pilot speed (job 43192997, 64 cores): comment file of 26.9M rows took 17 min, about 420 rows per second per core. That extrapolates to about 20,000 core-hours for the full archive before optimisation. Next: speed up hashing (48 hashes as 12x4 bands, simpler cleanup, MinHash in Spark SQL), timed on a fixed 1M-row slice.
- Pilot matches (Jaccard >= 0.7, other accounts): 427 in the comment file for 9 texts, 2 in the post file. All were created after the accused text, so none count under the earlier-only rule.

## Scope set by Emily (2026-11-12, later)
- Texts searched: comments and posts of the 987 accounts created **before that account's own accusation** (first_mention_adj_utc), >= 140 characters after stripping the game-post notice: 16,312 texts from 351 accounts (11,138 comments, 5,174 posts). All 987 accounts were accused before 30 Nov 2022 (Feb 2015 to 29 Nov 2022).
- A match counts only if the archive item was created **before that specific text** and **before 30 Nov 2022**.
- Archive rows dated 30 Nov 2022 or later (after converting CreatedAt to UTC) are dropped before hashing; the newer file batch (2025 onward) is skipped entirely.

- **Links (2026-11-12):** the 140-character minimum is measured after removing links (bare URLs dropped, markdown links reduced to their visible words), and links may be at most half of a text's characters. Applies to both sides. In scope this leaves 15,866 texts (10,830 comments, 5,036 posts) from 341 accounts.
