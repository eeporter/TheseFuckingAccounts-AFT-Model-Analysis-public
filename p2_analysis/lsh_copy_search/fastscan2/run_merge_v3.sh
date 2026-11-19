#!/usr/bin/env bash
# After fix_v3 succeeds: replace tasks 1, 3, 22 in full_out_v3/ with the rerun. Stops without changing anything unless
# all 3 rerun summaries say DONE with every self-check ok. Then deletes ONLY those 3 tasks' old partial files in
# full_out_v3/ and moves the rerun's files in. Counts only.
# Submit: sbatch --dependency=afterok:<fix_v3 job id> -A comdata -p cpu-g2 -c 1 --mem 4G --time 0:30:00 -J merge_v3 \
#           -o /gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2/logs/merge_v3_%j.out run_merge_v3.sh
# Job: 43360111 (afterok on fix_v3 array 43360110: runs only if all three rerun tasks exited 0).
# Order of safety checks: (1) afterok; (2) the Python below reads summary_task0001/0003/0022.json in full_out_v3_fix/
#   and exits non-zero unless every one says DONE and has a non-empty self-check list with all ok, and it checks all
#   three before the loop starts; (3) set -e stops the script there, so nothing in full_out_v3/ is touched.
# Only then, per task: print old/new file counts, delete full_out_v3/pairs_taskNNNN_*.parquet and summary_taskNNNN.json
#   for that task (the partial output of the failed run), and move the rerun files in. No other task is touched.
# Afterwards run_checks_v3b.sh (43360112) checks the merged folder.
set -eu
D=/gscratch/comdata/users/eeporter/reddit_copy_check/fastscan2
SIF=/gscratch/comdata/containers/cdsc_base/cdsc_base.sif
# apptainer bind: the host folder /mmfs1/gscratch is mounted at /gscratch inside the container, so the /gscratch paths
# used here resolve inside it (the first toy job, 43301873, failed because the container could not see /gscratch).
B="--bind /mmfs1/gscratch:/gscratch"
NEW=$D/full_out_v3_fix; MAIN=$D/full_out_v3
apptainer exec $B $SIF python3 - $NEW <<'PY'
import json, sys
for t in (1, 3, 22):
    T = json.load(open(f'{sys.argv[1]}/summary_task{t:04d}.json'))
    ok = T.get('DONE') and T.get('selfcheck') and all(x['ok'] for x in T['selfcheck'])
    print(json.dumps({'task': t, 'DONE': T.get('DONE'), 'selfcheck_ok': bool(ok), 'files': T.get('files_this_task'), 'rows': T.get('rows'),
                      'pairs': T.get('pairs'), 'pair_files': T.get('pair_files'), 'sink_pair_files': T.get('sink_pair_files'),
                      'n_big_row_groups': len(T.get('big_row_groups', [])), 'max_rss_one_worker_GB': T.get('max_rss_one_worker_GB')}))
    if not ok: sys.exit(f'task {t} not complete; nothing merged')
PY
# reached only if all three reruns passed the checks above
for t in 0001 0003 0022; do
  echo "task $t: old files in full_out_v3: $(ls $MAIN | grep -c "^pairs_task${t}_" || true); new files: $(ls $NEW | grep -c "^pairs_task${t}_" || true)"
  rm -f $MAIN/pairs_task${t}_*.parquet $MAIN/summary_task${t}.json
  mv $NEW/pairs_task${t}_*.parquet $NEW/summary_task${t}.json $MAIN/
done
echo "== merged; summaries now in full_out_v3: $(ls $MAIN/summary_task*.json | wc -l)"
