#!/usr/bin/env python3
"""Toy run of the LSH copy search (2026-11-12): toy_cands_v1.parquet against the archive file(s) in toy_files_v1.json.
Text -> lower case, URLs out, punctuation to spaces -> word 3-gram shingles -> HashingTF -> Spark MinHashLSH (100 hashes)
-> 20 bands of 5 hashes. Texts of 3-7 words skip LSH and match on an exact hash of the normalised text (band -1).
Pass 1 joins archive band keys to the broadcast candidate band keys. Pass 2 re-reads only matched rows and plant sources
and computes exact Jaccard on the shingle sets. Writes ids, metadata and scores only (no text)."""
import json, os, time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import NGram, HashingTF, MinHashLSH
from pyspark.ml.functions import vector_to_array

HERE = '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh'
OUT = f'{HERE}/toy_out_v1'
B, R, MINTOK, SHORTMIN, NF = 20, 5, 8, 3, 1 << 28
spark = SparkSession.builder.appName('lsh_toy_v1').getOrCreate(); spark.sparkContext.setLogLevel('WARN')
files = json.load(open(f'{HERE}/toy_files_v1.json'))['archive_files']
T = {}; t0 = time.time()

def norm(c):
    c = F.regexp_replace(F.lower(c), r'https?://\S+', ' ')
    c = F.regexp_replace(c, r"(?U)[^\w\s']", ' ')
    return F.trim(F.regexp_replace(c, r'(?U)\s+', ' '))
def prep(df):
    df = df.withColumn('norm', norm(F.col('text'))).drop('text').withColumn('tok', F.split('norm', ' '))
    df = df.withColumn('ntok', F.when(F.col('norm') == '', 0).otherwise(F.size('tok')))
    return NGram(n=3, inputCol='tok', outputCol='ng').transform(df.filter(F.col('ntok') >= SHORTMIN))
def vec(df):
    df = df.withColumn('ng2', F.when(F.col('ntok') >= MINTOK, F.col('ng')).otherwise(F.array(F.lit('__short__'))))
    return HashingTF(numFeatures=NF, binary=True, inputCol='ng2', outputCol='vec').transform(df)
def bands(df, model):
    df = model.transform(df).withColumn('sig', F.transform('hashes', lambda v: vector_to_array(v)[0]))
    lsh = F.transform(F.sequence(F.lit(0), F.lit(B - 1)),
                      lambda i: F.struct(i.alias('band'), F.xxhash64(i, F.slice('sig', i * R + 1, F.lit(R))).alias('key')))
    short = F.array(F.struct(F.lit(-1).alias('band'), F.xxhash64('norm').alias('key')))
    return df.withColumn('bk', F.when(F.col('ntok') >= MINTOK, lsh).otherwise(short))
def shingles(df): return df.withColumn('sh', F.array_distinct(F.transform('ng', lambda x: F.xxhash64(x))))
jac = lambda a, b: F.size(F.array_intersect(a, b)) / F.size(F.array_union(a, b))

cands = spark.read.parquet(f'{HERE}/toy_cands_v1.parquet')
cv = vec(prep(cands))
model = MinHashLSH(inputCol='vec', outputCol='hashes', numHashTables=B * R, seed=42).fit(cv)
cb = bands(cv, model).select('cix', F.explode('bk').alias('b')).select('cix', 'b.band', 'b.key').cache()
T['cand_band_rows'] = cb.count()

arch = (spark.read.parquet(*files).select(F.col('id').alias('aid'), F.col('author').alias('a_author'),
        F.col('subreddit').alias('a_subreddit'), F.col('CreatedAt').alias('a_created'), F.col('link_id').alias('a_link_id'),
        F.col('body').alias('text'))
        .filter(~F.trim(F.coalesce('text', F.lit(''))).isin('', '[deleted]', '[removed]')))
# pass 1: band join
ab = bands(vec(prep(arch.select('aid', 'text'))), model).select('aid', F.explode('bk').alias('b')).select('aid', 'b.band', 'b.key')
pairs = (ab.join(F.broadcast(cb), ['band', 'key']).groupBy('cix', 'aid')
         .agg(F.count('*').alias('n_bands'), F.max((F.col('band') == -1).cast('int')).alias('short_exact')))
pairs.write.mode('overwrite').parquet(f'{OUT}/pairs_raw'); T['pass1_s'] = round(time.time() - t0)
pairs = spark.read.parquet(f'{OUT}/pairs_raw'); T['raw_pairs'] = pairs.count()
T['archive_rows'] = spark.read.parquet(*files).count()

# pass 2: exact Jaccard for matched rows and for plant sources
t1 = time.time()
need = pairs.select('aid').union(cands.select(F.col('src_id').alias('aid')).dropna()).distinct()
a2 = shingles(prep(arch.join(F.broadcast(need), 'aid', 'left_semi'))).select(
    'aid', 'a_author', 'a_subreddit', 'a_created', 'a_link_id', F.col('ntok').alias('a_ntok'), F.col('sh').alias('a_sh')).cache()
c2 = shingles(prep(cands)).select('cix', F.col('id').alias('cid'), 'variant', 'src_id', F.col('ntok').alias('c_ntok'),
                                  F.col('sh').alias('c_sh')).cache()
out = (pairs.join(c2, 'cix').join(a2, 'aid')
       .withColumn('jaccard', jac('c_sh', 'a_sh')).withColumn('is_src', F.col('aid') == F.col('src_id')).drop('c_sh', 'a_sh'))
out.write.mode('overwrite').parquet(f'{OUT}/pairs'); out = spark.read.parquet(f'{OUT}/pairs')
# per-text summary (same format planned for the full copypasta candidate set); self/source rows excluded from counts
m = out.filter(~F.col('is_src') & (F.col('jaccard') >= 0.5))
per = (c2.select('cix', 'cid', 'variant').join(
        m.groupBy('cix').agg(F.count('*').alias('dupes_j50'), F.sum((F.col('jaccard') >= 0.9).cast('int')).alias('dupes_j90'),
                             F.countDistinct('a_author').alias('n_authors'), F.countDistinct('a_subreddit').alias('n_subreddits'),
                             F.min('a_created').alias('first_seen'), F.max('a_created').alias('last_seen')), 'cix', 'left')
       .fillna({'dupes_j50': 0, 'dupes_j90': 0, 'n_authors': 0, 'n_subreddits': 0}))
per.write.mode('overwrite').parquet(f'{OUT}/per_text_summary')
truth = (c2.filter(F.col('src_id').isNotNull()).join(a2.withColumnRenamed('aid', 'src_id'), 'src_id')
         .withColumn('true_jaccard', jac('c_sh', 'a_sh')).select('cix', 'variant', 'src_id', 'c_ntok', 'true_jaccard')
         .join(out.filter('is_src').select('cix', 'n_bands', 'short_exact').withColumn('found', F.lit(True)), 'cix', 'left')
         .fillna({'found': False, 'n_bands': 0})
         .withColumn('p_expected', F.when(F.col('c_ntok') < MINTOK, (F.col('true_jaccard') == 1).cast('double'))
                     .otherwise(1 - F.pow(1 - F.pow('true_jaccard', R), B))))
truth.write.mode('overwrite').parquet(f'{OUT}/plant_eval'); T['pass2_s'] = round(time.time() - t1)

# summaries (counts only)
ev = spark.read.parquet(f'{OUT}/plant_eval').toPandas()
by = ev.groupby('variant').agg(n=('cix', 'size'), found=('found', 'sum'), mean_true_j=('true_jaccard', 'mean'),
                               expected=('p_expected', 'sum')).round(2)
op = out.toPandas()
neg = op[op.variant == 'negative']
other = op[~op.is_src]
summ = {'timing_and_sizes': T, 'by_variant': by.reset_index().to_dict('records'),
        'missing_sources': int(ev.cix.nunique()) , 'candidates_total': cands.count(),
        'negative_pairs': int(len(neg)), 'negative_pairs_j_ge_0.5': int((neg.jaccard >= .5).sum()),
        'non_source_pairs': int(len(other)), 'non_source_pairs_j_ge_0.5': int((other.jaccard >= .5).sum()),
        'non_source_j_hist': other.jaccard.round(1).value_counts().sort_index().to_dict(),
        'pairs_per_cand_max': int(op.groupby('cix').size().max()) if len(op) else 0}
json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
