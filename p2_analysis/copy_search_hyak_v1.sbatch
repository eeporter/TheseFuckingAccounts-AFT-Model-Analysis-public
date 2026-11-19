#!/usr/bin/env bash
# Older copy search (12 Nov 2026): Slurm array that runs copy_search_hyak_v1.py (Aho-Corasick substring search of
# the search-list comments over the CDSC Reddit comments archive). Used for Hyak array job 43145502 (tasks 0-139).
# Each task gets 1 CPU and 8 GB and handles every 140th archive file (TASK = array index, NTASKS = 140, matching
# --array=0-139). ckpt-g2 is a preemptible checkpoint partition; --requeue restarts a preempted task, and the Python
# script skips archive files whose hits_parts/ output already exists, so it resumes.
# Output: hits_parts/file_<k>.parquet (combined afterwards by copy_search_combine_hyak_v1.py); logs in logs/.
#SBATCH --job-name=copy_search
#SBATCH --account=comdata
#SBATCH --partition=ckpt-g2
#SBATCH --array=0-139
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=logs/task_%a_%A.out
# environment: user bashrc, the project folder on /gscratch and its Python venv
source ~/.bashrc
cd /gscratch/comdata/users/eeporter/reddit_copy_check
source venv/bin/activate
# arguments: TASK = this array index, NTASKS = 140
python3 copy_search_hyak_v1.py "$SLURM_ARRAY_TASK_ID" 140
