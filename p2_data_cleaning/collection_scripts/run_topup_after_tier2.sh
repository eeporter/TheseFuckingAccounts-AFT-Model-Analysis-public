#!/bin/bash
# Waits until the second Tier 2 pass has finished, then runs the Tier 2 top-up.
#
S="$(cd "$(dirname "$0")" && pwd)"   # collection_scripts/: the folder holding this runner and the scripts it starts
while pgrep -f "tier2_pass2.py" >/dev/null; do sleep 60; done
echo "=== tier2 finished, TOP-UP started $(date '+%F %H:%M') ==="
cd /home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning || exit 1
python3 -u "$S/tier2_topup.py"
echo "=== TOP-UP exited rc=$? at $(date '+%F %H:%M') ==="
