#!/usr/bin/env bash
# 第三轮小补充（2026-09-27）：FAISS two_level_clustering 补种子 3/4，使它能以五种子进入主表（8 集 + MS MARCO；node0 串行）。
# 用法: setsid nohup bash chain_rev2c.sh > $ROOT/logs/rev2c.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 0 >/dev/null
rm -f $RES/msmarco/.rev_train_done
if ! pgrep -f chain_msmarco_eval_gpu.sh >/dev/null; then
  (setsid nohup bash chain_msmarco_eval_gpu.sh >> $ROOT/logs/msmarco_eval_gpu.log 2>&1 < /dev/null &)
  echo "restarted chain_msmarco_eval_gpu.sh $(date)"
fi
pgrep -f chain_eval_gpu.sh >/dev/null || echo "WARNING: chain_eval_gpu.sh is not running"
for ds in sift glove200 yahoo gist mxbai dbpedia wiki deep10m msmarco; do
  for s in 3 4; do
    $PIN $PY faiss_2lc.py $ds $s 2>&1 | grep --line-buffered -vE "^$"
  done
done
touch $RES/msmarco/.rev_train_done
echo "REV2C_DONE $(date)"
