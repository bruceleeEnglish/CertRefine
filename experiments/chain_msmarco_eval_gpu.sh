#!/usr/bin/env bash
# MS MARCO 的 GPU 稠密评测守护进程（GPU 0 + node1，不计时）：轮询 results/msmarco，直到补充实验训练完成且全部评测完。
set -uo pipefail
cd "$(dirname "$0")/msmarco"
source ../env.sh 1 >/dev/null
export CUDA_VISIBLE_DEVICES=0
while true; do
  $PIN $PY_ENC eval_ir_gpu.py --min-age 60 --device cuda:0 2>&1 | grep --line-buffered -vE "nothing to evaluate|^$"
  if [ -f $RES/msmarco/.rev_train_done ]; then
    left=$($PY_ENC -c "
import glob, json, os
od='$RES/msmarco'; n=0
for f in glob.glob(od+'/*.centroids.*'):
    nm=os.path.basename(f).split('.centroids')[0]; jp=os.path.join(od, nm+'.ir.json')
    n += not (os.path.exists(jp) and json.load(open(jp)).get('evaluator')=='gpu-dense-ir-v1')
print(n)")
    [ "$left" -eq 0 ] && break
  fi
  sleep 120
done
echo "MSMARCO_GPU_EVAL_DONE $(date)"
