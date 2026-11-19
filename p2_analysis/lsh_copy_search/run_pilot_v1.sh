#!/usr/bin/env bash
# Pilot: full copypasta candidate set (>=140 chars) vs one submission file + one comment file. 64 cores on cpu-g2.
set -eu
L=/gscratch/comdata/users/eeporter/reddit_copy_check/lsh
export SPARK_CONF_DIR=$L/conf PYSPARK_PYTHON=python3
cd $L
echo "== pilot start $(date +%T) on $(hostname)"
/gscratch/comdata/containers/bin/spark-submit --master 'local[64]' $L/lsh_pilot_v1.py 256
echo "== pilot end $(date +%T)"
