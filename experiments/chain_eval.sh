#!/usr/bin/env bash
# 评测守护进程（NUMA node1，不计时）：轮询各数据集结果目录，对新出现的质心文件做等扫描评测。
# 用法: setsid nohup bash chain_eval.sh > $ROOT/logs/eval.log 2>&1 < /dev/null &
# 停止：touch $RES/.eval_stop
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 1 >/dev/null
while [ ! -f $RES/.eval_stop ]; do
  for dir in $RES/*/; do
    ds=$(basename $dir)
    case $ds in msmarco) continue;; esac
    base=${ds%%_c*}; c=${ds#*_c}; [ "$c" = "$ds" ] && c=""
    if [ -n "$c" ]; then export SCDKM_C=$c; else unset SCDKM_C; fi
    $PIN $PY eval_scan.py $base --min-age 30 2>&1 | grep -vE "nothing to evaluate|^$"
  done
  unset SCDKM_C
  sleep 60
done
echo "EVAL_STOPPED $(date)"
