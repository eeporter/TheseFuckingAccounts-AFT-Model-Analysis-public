#!/bin/bash
# Gap fill: reruns Tier 3 and then Tier 4 for accounts added after the first pass (both scripts skip finished accounts).
# Tier 4 only starts if Tier 3 exits cleanly.
S="$(cd "$(dirname "$0")" && pwd)"   # collection_scripts/: the folder holding this runner and the scripts it starts
cd /home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning || exit 1
echo "=== TIER 3 gap started $(date '+%F %H:%M') ==="
python3 -u "$S/tier3_arctic_account_meta.py"; rc=$?
echo "=== TIER 3 exited rc=$rc at $(date '+%F %H:%M') ==="
[ $rc -ne 0 ] && { echo "TIER 3 FAILED - not starting Tier 4"; exit $rc; }
echo "=== TIER 4 gap started $(date '+%F %H:%M') (fixed cursor, cap 10000/stream) ==="
python3 -u "$S/tier4_arctic_full_history.py"
echo "=== TIER 4 exited rc=$? at $(date '+%F %H:%M') ==="
