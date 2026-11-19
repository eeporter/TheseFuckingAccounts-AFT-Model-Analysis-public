#!/usr/bin/env bash
# Toy run: build toy set, then Spark LSH job. Spark shuffle/spill goes to node-local /tmp, not the 96%-full scrubbed.
set -eu
L=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
mkdir -p $L/conf $L/logs
cp /gscratch/comdata/local/spark_config/spark-defaults.conf /gscratch/comdata/local/spark_config/log4j.properties.template $L/conf/ 2>/dev/null || true
sed "s#^SPARK_LOCAL_DIRS=.*#SPARK_LOCAL_DIRS=/tmp/eeporter_spark_\${SLURM_JOB_ID:-x}#; s#^SPARK_WORKER_DIR=.*#SPARK_WORKER_DIR=/tmp/eeporter_spark_work#" \
    /gscratch/comdata/local/spark_config/spark-env.sh > $L/conf/spark-env.sh
export SPARK_CONF_DIR=$L/conf PYSPARK_PYTHON=python3
cd $L
echo "== make_toy $(date +%T)"; apptainer exec $SIF python3 $L/make_toy_v1.py
echo "== archive CreatedAt type"; apptainer exec $SIF python3 -c "import glob,pyarrow.parquet as pq; f=sorted(glob.glob('/gscratch/comdata/output/reddit_comments_by_author.parquet/*.parquet'))[0]; print(pq.ParquetFile(f).schema_arrow.field('CreatedAt'), pq.ParquetFile(f).schema.column(3))"
echo "== spark $(date +%T)"; /gscratch/comdata/containers/bin/spark-submit --master 'local[4]' $L/lsh_toy_v1.py
echo "== done $(date +%T)"
