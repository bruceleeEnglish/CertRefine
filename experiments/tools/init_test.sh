#!/usr/bin/env bash
# 起步分配方式的计时与一致率对照（node1；暂停 node1 上的评测进程，测完恢复）。
# 用法: bash tools/init_test.sh <script.py> <ds...>
set -uo pipefail
cd /path/to/workspace
source experiments/env.sh 1 >/dev/null
SCRIPT=$1; shift
export RES=$ROOT/results_test
P=$(pgrep -u user -f "[b]in/python eval_scan.py" | tr "\n" " ")
echo "pausing eval: $P"
[ -n "$P" ] && kill -STOP $P
trap '[ -n "$P" ] && kill -CONT $P && echo "resumed eval"' EXIT
for DS in "$@"; do
  mkdir -p $RES/$DS; cp -n $ROOT/results/$DS/hier_s42.* $RES/$DS/ 2>/dev/null
  for m in 4 8; do
    echo "=== $DS ivfgemm m=$m"
    $PIN $PY experiments/$SCRIPT $DS --seed 42 --tag ig$m --init ivfgemm --ann-nprobe $m --passes 5 --save-every 0 --diag-exact 2>&1 | grep -E "^diag|^init|Error|Traceback|epoch.: 5," | sed -E "s/'pairs_total.*'total_s/ total_s/; s/, 'empty.*//"
  done
done
