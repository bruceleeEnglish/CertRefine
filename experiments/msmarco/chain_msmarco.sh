#!/usr/bin/env bash
# MS MARCO 数据准备与评测（NUMA node1 + 两张 GPU；不计时）。
# 1) 等语料下载完成 → 2) 两张卡并行编码语料分片 → 3) 编码 dev 查询 → 4) 合并、精确 top-100、qrels
# 5) 之后轮询 results/msmarco，对新质心做端到端检索评测（eval_ir.py），直到 .train_done 且全部评测完。
set -uo pipefail
cd "$(dirname "$0")"
source ../env.sh 1 >/dev/null
export HF_HOME=$DATA_MSMARCO/hf_cache
RAW=$DATA_MSMARCO/raw
until [ -f $RAW/corpus.parquet ] && grep -q DL_OK $ROOT/logs/dl_all.log 2>/dev/null; do sleep 60; done
echo "corpus ready $(date)"
bash get_model.sh $DATA_MSMARCO/bge-small-en-v1.5 > $ROOT/logs/get_model.log 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ ! -f $DATA_MSMARCO/msmarco_meta.json ]; then
  $PIN $PY_ENC prepare.py encode --shard 0 --nshards 2 --device cuda:0 > $ROOT/logs/enc0.log 2>&1 &
  P0=$!
  sleep 60   # 错开两个进程读 parquet 的内存高峰
  $PIN $PY_ENC prepare.py encode --shard 1 --nshards 2 --device cuda:1 > $ROOT/logs/enc1.log 2>&1 &
  P1=$!
  wait $P0 $P1
  echo "encode done $(date)"
  $PIN $PY_ENC prepare.py queries --device cuda:0
  $PIN $PY_ENC prepare.py finalize --device cuda:0
fi
echo "msmarco data ready $(date)"
while true; do
  $PIN $PY eval_ir.py 2>&1 | grep --line-buffered -vE "nothing to evaluate|^$"
  if [ -f $RES/msmarco/.train_done ]; then
    left=$(ls $RES/msmarco/*.centroids.* 2>/dev/null | while read f; do b=$(basename $f); nm=${b%%.centroids*}; [ -f $RES/msmarco/$nm.ir.json ] || echo $nm; done | wc -l)
    [ "$left" -eq 0 ] && break
  fi
  sleep 120
done
echo "MSMARCO_EVAL_DONE $(date)"
