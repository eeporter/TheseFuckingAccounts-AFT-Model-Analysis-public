#!/usr/bin/env python3
"""Toy test of the exact copy search (fastscan), 2026-11-17.
Inputs (same toy set the Spark toy run used, lsh/ folder on Hyak):
  toy_cands_v1.parquet     71 rows  (toy set: 10 sources x 6 variants, 3 short x 2, 5 negatives)
  toy_haystack_v1.parquet  5,000 rows (first 5,000 rows of archive comment file part-00000)
  toy_out_v1/plant_eval    66 rows  (Spark's exact Jaccard of each plant vs its source)
  toy_out_v1/pairs         44 rows  (pairs Spark LSH found)
Same rules as lsh_toy_v1.py: Spark cleanup, word 3-grams, >= 3 words, archive '', [deleted], [removed] dropped,
no 140-character cut (the toy predates it). Dupe = exact Jaccard >= 0.7.
Checks: (A) index result == brute force over all 71 x 5,000 pairs; (B) Jaccard equals Spark's for every plant;
(C) every Spark pair with J >= 0.7 is found. Prints counts only; writes ids + scores only (no text, no usernames).

Added notes (comment-only pass):
  Jobs: Hyak Slurm 43301873 (failed before reading data: the container could not see /gscratch; fixed with
    --bind /mmfs1/gscratch:/gscratch) and 43301955 (PASS: 33 pairs at J >= 0.7, same as brute force and Spark).
    Launched by run_toy_fast_v1.sh.
  Outputs (toy_fast_out_v1/ next to this file): pairs.parquet (ids + scores of pairs found) and summary.json.
  PASS = (A) and no plant's Jaccard differs from Spark's by > 1e-9 and (C) all Spark J >= 0.7 pairs found.
  Method: fastscan.Index (exact prefix filter, length filter, exact intersection); no Bloom filter in this version.
"""
import json, os, sys, time
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastscan as fs

L = os.environ.get('TOY_DIR', '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_fast_out_v1'); os.makedirs(OUT, exist_ok=True)
T = {}; t0 = time.time()

c = pq.read_table(f'{L}/toy_cands_v1.parquet', columns=['cix', 'variant', 'src_id', 'text'])
h = pq.read_table(f'{L}/toy_haystack_v1.parquet', columns=['id', 'body'])
pe = pq.read_table(f'{L}/toy_out_v1/plant_eval', columns=['cix', 'variant', 'src_id', 'true_jaccard']).to_pandas()
sp = pq.read_table(f'{L}/toy_out_v1/pairs', columns=['cix', 'aid', 'jaccard', 'is_src']).to_pandas()
T['rows'] = {'toy_cands': c.num_rows, 'haystack': h.num_rows, 'spark_plant_eval': len(pe), 'spark_pairs': len(sp)}

qn = fs.norm(c['text'])
body = pc.fill_null(h['body'], '')
# haystack rows that are empty, [deleted] or [removed] (after trimming spaces) are skipped, as in the Spark toy
keep = ~np.isin(pc.utf8_trim(body, ' ').to_numpy(zero_copy_only=False), ['', '[deleted]', '[removed]'])
an = fs.norm(body)
T['haystack_rows_kept'] = int(keep.sum())

t1 = time.time()
idx = fs.Index(qn, 7, 10)
r, q, inter, m, j = idx.scan(an, keep)
T['scan_s'] = round(time.time() - t1, 3)
T['index'] = {'queries_with_shingles': int(idx.ok.sum()), 'query_shingles': idx.n_shingles,
              'prefix_entries': idx.n_prefix, 'rows_shingled': int(idx.stats[0]), 'pairs_verified': int(idx.stats[1])}
cix = c['cix'].to_numpy(); aid = h['id'].to_numpy(zero_copy_only=False)
fast = {(int(cix[a]), aid[b]) for a, b in zip(q, r)}
T['fast_pairs_j70'] = len(fast)

# (A) brute force over every pair, exact integer test 10c >= 7(|A|+|B|-c)
# Brute-force checker: shingle every query and every kept haystack row (same shingles as the index uses), then
# test all query x haystack pairs with Python sets, no index and no filters. Jsrc = Jaccard of each plant
# against its own source row in the haystack (when the source is there), used for check (B).
qs, qp, _ = fs.shingles(qn); hs, hp, _ = fs.shingles(an)
Q = [set(qs[qp[i]:qp[i + 1]].tolist()) for i in range(len(qp) - 1)]
H = [set(hs[hp[k]:hp[k + 1]].tolist()) if keep[k] else set() for k in range(len(hp) - 1)]
bf, Jsrc = set(), {}
pos = {v: k for k, v in enumerate(aid)}
src = c['src_id'].to_pylist()
for i, A in enumerate(Q):
    if not A: continue
    for k, B in enumerate(H):
        if not B: continue
        x = len(A & B)
        if x and 10 * x >= 7 * (len(A) + len(B) - x): bf.add((int(cix[i]), aid[k]))
    if src[i] is not None and src[i] in pos:
        B = H[pos[src[i]]]; x = len(A & B); Jsrc[int(cix[i])] = x / (len(A) + len(B) - x) if (A or B) else float('nan')
T['A_bruteforce_pairs_j70'] = len(bf); T['A_index_equals_bruteforce'] = fast == bf

# (B) plant Jaccard vs Spark's exact Jaccard
pe['fast_j'] = pe.cix.map(Jsrc)
pe['diff'] = (pe.fast_j - pe.true_jaccard).abs()
pe['spark_ge70'] = pe.true_jaccard >= 0.7
pe['fast_found'] = [(int(a), s) in fast for a, s in zip(pe.cix, pe.src_id)]
T['B_plant_max_abs_jaccard_diff'] = float(pe['diff'].max())
T['B_plants_jaccard_differs'] = int((pe['diff'] > 1e-9).sum())
T['B_by_variant'] = (pe.groupby('variant').agg(n=('cix', 'size'), spark_true_j_ge70=('spark_ge70', 'sum'),
                     fast_found=('fast_found', 'sum')).reset_index().to_dict('records'))

# (C) Spark LSH pairs with J >= 0.7 all found?
s70 = sp[sp.jaccard >= 0.7]
sset = {(int(a), b) for a, b in zip(s70.cix, s70.aid)}
T['C_spark_pairs_j70'] = len(sset); T['C_found_by_fast'] = len(sset & fast)
T['C_fast_only_pairs'] = len(fast - sset)        # pairs LSH missed (or were never banded)
spj = {(int(a), b): v for a, b, v in zip(sp.cix, sp.aid, sp.jaccard)}
fj = {(int(cix[a]), aid[b]): v for a, b, v in zip(q, r, j)}
T['C_max_abs_jaccard_diff_on_shared_pairs'] = max([abs(fj[k] - spj[k]) for k in sset & fast] or [0.0])

pq.write_table(pa.table({'cix': cix[q], 'aid': aid[r], 'inter': inter, 'a_nsh': m, 'q_nsh': idx.qlen[q], 'jaccard': j,
                         'is_src': [s == a for s, a in zip(np.array(src, dtype=object)[q], aid[r])]}), f'{OUT}/pairs.parquet')
T['total_s'] = round(time.time() - t0, 2)
T['PASS'] = bool(T['A_index_equals_bruteforce'] and T['B_plants_jaccard_differs'] == 0 and T['C_found_by_fast'] == T['C_spark_pairs_j70'])
json.dump(T, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
print(json.dumps(T, indent=1, default=str))
