#!/bin/bash
# First full run: Tier 3 (account meta), then Tier 4 (full histories) if Tier 3 exits cleanly.
S="$(cd "$(dirname "$0")" && pwd)"   # collection_scripts/: the folder holding this runner and the scripts it starts
cd /home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning || exit 1
echo "=== TIER 3 started $(date '+%F %H:%M') ==="
python3 -u "$S/tier3_arctic_account_meta.py"
rc=$?
echo "=== TIER 3 exited rc=$rc at $(date '+%F %H:%M') ==="
if [ $rc -ne 0 ]; then echo "TIER 3 FAILED - not starting Tier 4"; exit $rc; fi
echo "=== TIER 4 started $(date '+%F %H:%M') ==="
python3 -u "$S/tier4_arctic_full_history.py"
echo "=== TIER 4 exited rc=$? at $(date '+%F %H:%M') ==="
