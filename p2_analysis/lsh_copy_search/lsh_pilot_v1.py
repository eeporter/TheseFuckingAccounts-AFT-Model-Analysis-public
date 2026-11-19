#!/usr/bin/env python3
"""LSH copypasta search, pilot (2026-11-12). Full copypasta candidate set (texts >= 140 characters) against one archive
submission file and one archive comment file, one file at a time, results written after each file.
Settings agreed with Emily: texts under 140 characters dropped on both sides; lower case, URLs out, punctuation to
spaces; word 3-gram shingles; HashingTF 2^28 -> Spark MinHashLSH 100 hashes -> 20 bands x 5; dupe = exact Jaccard >= 0.7.
The candidate band keys are broadcast; the archive side is streamed (repartitioned locally for parallelism, never joined
in full). Pass 2 re-reads only matched archive rows and computes exact Jaccard. Outputs ids, metadata and scores only."""
import glob, json, sys, time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import NGram, HashingTF, MinHashLSH
from pyspark.ml.functions import vector_to_array

HERE = '/gscratch/comdata/users/eeporter/reddit_copy_check/lsh'
OUT = f'{HERE}/pilot_out_v1'
B, R, NF, MINCHAR, DUPE, PARTS = 20, 5, 1 << 28, 140, 0.7, int(sys.argv[1]) if len(sys.argv) > 1 else 256
SUB = sorted(glob.glob('/gscratch/comdata/output/reddit_submissions_by_author.parquet/*.parquet'))[0]
COM = sorted(glob.glob('/gscratch/comdata/output/reddit_comments_by_author.parquet/*.parquet'))[1]
spark = SparkSession.builder.appName('lsh_pilot_v1').getOrCreate()

def norm(c):
    c = F.regexp_replace(F.lower(c), r'https?://\S+', ' ')
    c = F.regexp_replace(c, r"(?U)[^\w\s']", ' ')
    return F.trim(F.regexp_replace(c, r'(?U)\s+', ' '))
def prep(df):   # df has text; keeps >= 140 raw characters and >= 3 words (so there is at least one shingle)
    df = df.filter(F.length(F.trim('text')) >= MINCHAR).withColumn('norm', norm(F.col('text'))).drop('text')
    df = df.withColumn('tok', F.split('norm', ' ')).filter((F.col('norm') != '') & (F.size('tok') >= 3))
    return NGram(n=3, inputCol='tok', outputCol='ng').transform(df)
tf = HashingTF(numFeatures=NF, binary=True, inputCol='ng', outputCol='vec')
def bands(df, model):
    df = model.transform(tf.transform(df)).withColumn('sig', F.transform('hashes', lambda v: vector_to_array(v)[0]))
    return df.withColumn('bk', F.transform(F.sequence(F.lit(0), F.lit(B - 1)), lambda i: F.struct(
        i.alias('band'), F.xxhash64(i, F.slice('sig', i * R + 1, F.lit(R))).alias('key'))))
def shingles(df): return df.withColumn('sh', F.array_distinct(F.transform('ng', lambda x: F.xxhash64(x))))
jac = lambda a, b: F.size(F.array_intersect(a, b)) / F.size(F.array_union(a, b))

# candidate side (small): key = kind:id
c = (spark.read.parquet(f'{HERE}/candidates_v1.parquet')
     .select(F.concat_ws(':', 'kind', 'id').alias('ckey'), F.col('id').alias('cid'), F.col('kind').alias('ckind'),
             F.col('author').alias('c_author'), 'text'))
cp = prep(c).cache()
model = MinHashLSH(inputCol='vec', outputCol='hashes', numHashTables=B * R, seed=42).fit(tf.transform(cp))
cb = bands(cp, model).select('ckey', F.explode('bk').alias('b')).select('ckey', 'b.band', 'b.key').cache()
c2 = shingles(cp).select('ckey', 'cid', 'ckind', 'c_author', F.size('tok').alias('c_ntok'), F.col('sh').alias('c_sh')).cache()
meta = {'candidates_ge140': c2.count(), 'candidate_band_rows': cb.count()}
print('PILOT candidates', meta, flush=True)

def archive(kind, path):
    d = spark.read.parquet(path)
    text = F.col('body') if kind == 'comment' else F.concat_ws('\n', F.coalesce('title', F.lit('')), F.coalesce('selftext', F.lit('')))
    link = F.col('link_id') if kind == 'comment' else F.lit(None).cast('string')
    return d.select(F.col('id').alias('aid'), F.lit(kind).alias('akind'), F.col('author').alias('a_author'),
                    F.col('subreddit').alias('a_subreddit'), F.col('CreatedAt').alias('a_created'), link.alias('a_link_id'),
                    text.alias('text'))

for kind, path in [('post', SUB), ('comment', COM)]:
    o = f'{OUT}/{kind}'; T = {'file': path}; t0 = time.time()
    a = archive(kind, path)
    T['archive_rows'] = a.count(); T['archive_rows_ge140'] = a.filter(F.length(F.trim('text')) >= MINCHAR).count()
    T['count_s'] = round(time.time() - t0); t1 = time.time()
    # pass 1: band join (archive rows repartitioned so all cores work; file has only ~20 row groups)
    ab = bands(prep(a.select('aid', 'text').filter(F.length(F.trim('text')) >= MINCHAR).repartition(PARTS)), model)
    pairs = (ab.select('aid', F.explode('bk').alias('b')).select('aid', 'b.band', 'b.key')
             .join(F.broadcast(cb), ['band', 'key']).groupBy('ckey', 'aid').agg(F.count('*').alias('n_bands')))
    pairs.write.mode('overwrite').parquet(f'{o}/pairs_raw'); pairs = spark.read.parquet(f'{o}/pairs_raw')
    T['raw_pairs'] = pairs.count(); T['pass1_s'] = round(time.time() - t1); t2 = time.time()
    # pass 2: exact Jaccard on matched rows only
    a2 = shingles(prep(a.join(F.broadcast(pairs.select('aid').distinct()), 'aid', 'left_semi'))).select(
        'aid', 'akind', 'a_author', 'a_subreddit', 'a_created', 'a_link_id', F.size('tok').alias('a_ntok'), F.col('sh').alias('a_sh'))
    out = (pairs.join(c2, 'ckey').join(a2, 'aid').withColumn('jaccard', jac('c_sh', 'a_sh'))
           .withColumn('is_self', (F.col('aid') == F.col('cid')) & (F.col('akind') == F.col('ckind')))
           .withColumn('same_author', F.col('a_author') == F.col('c_author')).drop('c_sh', 'a_sh'))
    out.write.mode('overwrite').parquet(f'{o}/pairs'); out = spark.read.parquet(f'{o}/pairs')
    d = out.filter(~F.col('is_self') & (F.col('jaccard') >= DUPE))
    per = (c2.select('ckey', 'cid', 'ckind').join(d.groupBy('ckey').agg(
               F.count('*').alias('dupes_j70'), F.sum((F.col('jaccard') >= 0.9).cast('int')).alias('dupes_j90'),
               F.sum((~F.col('same_author')).cast('int')).alias('dupes_other_author'),
               F.countDistinct('a_author').alias('n_authors'), F.countDistinct('a_subreddit').alias('n_subreddits'),
               F.min('a_created').alias('first_seen'), F.max('a_created').alias('last_seen')), 'ckey', 'left')
           .join(out.filter('is_self').select('ckey', F.lit(True).alias('self_found')), 'ckey', 'left')
           .fillna({'dupes_j70': 0, 'dupes_j90': 0, 'dupes_other_author': 0, 'n_authors': 0, 'n_subreddits': 0, 'self_found': False}))
    per.write.mode('overwrite').parquet(f'{o}/per_text_summary')
    T['pass2_s'] = round(time.time() - t2); T['total_s'] = round(time.time() - t0)
    T['pairs_scored'] = out.count(); T['pairs_j70_non_self'] = d.count(); T['self_matches'] = out.filter('is_self').count()
    T['self_matches_j_lt_1'] = out.filter('is_self and jaccard < 1').count()
    T['cands_with_dupes'] = per.filter('dupes_j70 > 0').count(); T['max_dupes_one_text'] = per.agg(F.max('dupes_j70')).first()[0]
    T['j_hist'] = {str(r[0]): r[1] for r in out.groupBy(F.round('jaccard', 1)).count().orderBy(1).collect()}
    json.dump({**meta, **T}, open(f'{o}/timing_summary.json', 'w'), indent=1, default=str)
    print('PILOT DONE', kind, json.dumps({**meta, **T}, default=str), flush=True)
