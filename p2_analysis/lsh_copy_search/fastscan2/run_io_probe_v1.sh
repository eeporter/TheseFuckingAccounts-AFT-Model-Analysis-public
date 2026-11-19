#!/usr/bin/env bash
# READ-ONLY disk test (io_probe_v1.py): reads CreatedAt+body column chunks of older-batch comment files from sorted position 600 on (stops before 620),
# arms of 4, 12, 24, 48 and again 4 reader processes, max(48, 2 x readers) row groups each (288 row groups, ~16 files,
# ~20 GB read in total). Writes one counts-only JSON, updated after every arm.
# Submit: sbatch -A comdata -p cpu-g2 -c 4 --mem 32G --time 0:20:00 -J io_probe_v1 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/io_probe_v1_%j.out run_io_probe_v1.sh
# Job: 43332089. Read-speed probe: does more reads in flight speed up reading the archive? Writes nothing else.
# Output: io_probe_v1/io_probe_v1_<job id>.json (MB/s per arm, speedup relative to the first arm).
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
mkdir -p $D/io_probe_v1
cd $D
echo "== host $(hostname) $(date +%T) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-x}"
apptainer exec $B $SIF python3 $D/io_probe_v1.py --offset 600 --rgs-per-arm 48 --arms 4,12,24,48,4 \
  --out $D/io_probe_v1/io_probe_v1_${SLURM_JOB_ID:-x}.json
echo "== done $(date +%T)"
