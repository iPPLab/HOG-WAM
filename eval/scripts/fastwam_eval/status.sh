#!/bin/bash
# Progress report for a FastWAM campaign. Reads only the result files, so it is correct
# while workers run and after any number of crashes.
#
# Usage: status.sh <run_tag> [--full]
set -u
TAG=${1:?run tag}
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/_paths.sh"
ROOT=$FW_ROOT
RES=$ROOT/results/$TAG
TASKS=${FW_TASKS:-$H/tasks/all.txt}
SEEDS=${TW_SEEDS:-100}
[ -d "$RES" ] || { echo "no result dir $RES"; exit 1; }

tot=0; done_n=0; suc=0; gvtot=0
printf "%-28s %7s %8s\n" task done rate
while IFS= read -r t; do
  [ -z "$t" ] && continue; case "$t" in \#*) continue ;; esac
  d=$(ls "$RES/$t"/*.json 2>/dev/null | wc -l)
  s=$(grep -l '"success":true' "$RES/$t"/*.json 2>/dev/null | wc -l)
  gv=$(ls "$RES/$t"/*.giveup 2>/dev/null | wc -l); gvtot=$((gvtot+gv))
  tot=$((tot+SEEDS)); done_n=$((done_n+d)); suc=$((suc+s))
  if [ "$d" -gt 0 ]; then
    printf "%-28s %3s/%-3s %7s%%\n" "$t" "$d" "$SEEDS" "$(awk -v a=$s -v b=$d 'BEGIN{printf "%.0f",100*a/b}')"
  elif [ "${2:-}" = "--full" ]; then
    printf "%-28s %3s/%-3s %8s\n" "$t" 0 "$SEEDS" "-"
  fi
done < "$TASKS"
echo
echo "progress: $done_n/$tot episodes ($(awk -v a=$done_n -v b=$tot 'BEGIN{printf "%.1f",100*a/b}')%)   successes $suc   overall $(awk -v a=$suc -v b=$done_n 'BEGIN{if(b)printf "%.1f%%",100*a/b; else print "-"}')"
[ "$gvtot" -gt 0 ] && echo "WARNING: $gvtot unit(s) given up after repeated infra failures -- check logs/, then delete the .giveup to retry"
echo "workers: $(ps -eo cmd -ww --no-headers | grep -v 'bash -c' | grep -c '[f]astwam_eval/worker\.sh')   sims: $(ps -eo cmd -ww --no-headers | grep -v 'bash -c' | grep -c '[e]val_policy_client')   held locks: $(find "$RES" -name '*.lock' -type d 2>/dev/null | wc -l)"
