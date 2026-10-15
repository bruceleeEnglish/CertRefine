#!/usr/bin/env bash
# chain_rev.sh 之后的补充计时（node0，串行）：线程扩展表里 20 线程这一档（带分阶段计时的新版 refine.py）。
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 0 >/dev/null
until grep -q "^REV_DONE" $ROOT/logs/rev.log 2>/dev/null; do sleep 60; done
for ds in sift gist; do
  dir=$RES/$ds
  [ -f $dir/cr_thr20_s42.refine.json ] && continue
  echo "--- refine $ds cr_thr20 $(date +%T)"
  $PIN $PY refine.py $ds --seed 42 --tag cr_thr20 --passes 5 --threads 20 --save-every -1 2>&1 | grep -E "^(init|saved|Traceback|.*Error)"
done
echo "REV_POST_DONE $(date)"
