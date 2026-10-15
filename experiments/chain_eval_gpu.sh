#!/usr/bin/env bash
# GPU 评测守护进程（取代 chain_eval.sh；GPU 1 + node1 CPU，不计时）：轮询各数据集结果目录，对新出现的质心做等扫描评测。
# 与 CPU 版逐格一致（eval_scan_gpu.py --verify 在 SIFT/GIST/Deep-10M 上验证过）。停止：touch $RES/.eval_stop
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 1 >/dev/null
export CUDA_VISIBLE_DEVICES=1
while [ ! -f $RES/.eval_stop ]; do
  for dir in $RES/*/; do
    ds=$(basename $dir)
    case $ds in msmarco) continue;; esac
    base=${ds%%_c*}; c=${ds#*_c}; [ "$c" = "$ds" ] && c=""
    if [ -n "$c" ]; then export SCDKM_C=$c; else unset SCDKM_C; fi
    $PIN $PY_ENC eval_scan_gpu.py $base --min-age 30 --device cuda:0 2>&1 | grep --line-buffered -vE "nothing to evaluate|^$"
  done
  unset SCDKM_C
  sleep 30
done
echo "EVAL_STOPPED $(date)"
