# fastscan: exact copy search (replaces Spark MinHash LSH)

Plan and cost estimates: [Fastest copy search plan](https://claude.ai/artifact/JThKSqqSoejV1SV78wq8fK) (16 Nov 2026).
CDSC rule: nothing here contains comment text or usernames. Logs are counts and IDs only.

## Method

- Cleanup is identical to `lsh_pilot_v1.py` (Spark): lower case, `https?://\S+` removed, `(?U)[^\w\s']` turned into spaces, whitespace collapsed, then trimmed. This is done with pyarrow.compute in `fastscan.norm`.
- Shingles are word 3-grams, hashed like Spark: xxhash64, seed 42, over the UTF-8 bytes of `"w1 w2 w3"`.
- **Exact prefix filter (no LSH).** Each query text indexes only its rarest `|A| - ceil(0.7|A|) + 1` 3-grams. Rarity is document frequency within the queries. Any archive text with Jaccard ≥ 0.7 must contain one of them. Candidates are length-filtered (0.7·|A| ≤ |B| ≤ |A|/0.7) and then verified by an exact set intersection. Nothing is approximate.
- The C core is `fastscan.c`, built with the system gcc on the compute node. The Python side is `fastscan.py`, run in the CDSC container `cdsc_base.sif` with `--bind /mmfs1/gscratch:/gscratch`.

## Files

| File | What |
| --- | --- |
| fastscan.c | xxhash64, 3-gram shingling, prefix index, scan + exact verify |
| fastscan.py | Spark-equivalent cleanup, index build, scan wrapper, brute-force checker |
| toy_fast_v1.py / run_toy_fast_v1.sh | toy test against the Spark toy run |
| pilot_fast_v1.py / run_pilot_fast_v1.sh | rerun of the Spark pilot comment file; checks all Spark matches come back |
| logs/ | Slurm logs and summary.json of each run (counts only) |

## Run log

| Date | Job | What | Result |
| --- | --- | --- | --- |
| 2026-11-17 | 43301873 | toy test, 2 cores | FAILED before reading data: the container could not see `/gscratch` (symlink). Fixed with `--bind /mmfs1/gscratch:/gscratch`. |
| 2026-11-17 | 43301955 | toy test, 2 cores, 8 s | **PASS.** 71 toy texts vs 5,000-row haystack. 33 pairs at J ≥ 0.7, the same as brute force over all 355,000 pairs and the same as Spark. Jaccard is identical for all 66 plants. No extra pairs and no negatives. |
| 2026-11-17 | 43321861 | pilot comment file, 8 cores, 43 s | **PASS.** Same file and rules as Spark pilot job 43192997 (26,922,660 rows; 36,089 queries ≥ 140 characters). Found all 427 Spark matches at J ≥ 0.7, with identical Jaccard and no extra pairs. 189 CPU-seconds in total (142,000 rows/s per core), against about 18 core-hours for Spark. Peak memory 6.1 GB. |

Toy command: `sbatch -A comdata -p cpu-g2 -c 2 --mem 8G --time 0:15:00 -J toy_fast_v1 -o .../fastscan/toy_fast_v1_%j.out run_toy_fast_v1.sh`

Pilot command: `sbatch -A comdata -p cpu-g2 -c 8 --mem 32G --time 0:30:00 -J pilot_fast_v1 -o .../fastscan/pilot_fast_v1_%j.out run_pilot_fast_v1.sh`
