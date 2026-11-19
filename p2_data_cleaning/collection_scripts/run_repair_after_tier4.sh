#!/bin/bash
# Waits until tier4_arctic_full_history.py has finished, then runs the cursor repair (repair_history_cursor.py) from p2_data_cleaning/.
S="$(cd "$(dirname "$0")" && pwd)"   # collection_scripts/: the folder holding this runner and the scripts it starts
while pgrep -f "tier4_arctic_full_history.py" >/dev/null; do sleep 60; done
echo "=== tier4 finished, REPAIR started $(date '+%F %H:%M') ==="
cd /home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning || exit 1
python3 -u "$S/repair_history_cursor.py"
echo "=== REPAIR exited rc=$? at $(date '+%F %H:%M') ==="
