#!/usr/bin/env bash
# 与 chain_extra.sh 的 voronoi 阶段完全相同的诊断运行（--diag-exact，不存质心，其计时不进入论文），
# 逆序放在 node1 上与 node0 的计时链并行，缩短总时长。chain_extra.sh 到达 voronoi 阶段时会跳过已完成的数据集。
# 用法: setsid nohup bash tools/voronoi_node1.sh > $ROOT/logs/voronoi_node1.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source ./env.sh 1 >/dev/null
for ds in deep10m wiki dbpedia mxbai gist yahoo glove200 sift; do
  dir=$($PY -c "import sys; sys.path.insert(0,'.'); from common import res_dir; print(res_dir('$ds'))")
  for tag in vor_cr vor_sl; do
    if [ -f $dir/${tag}_s42.refine.json ]; then echo "skip $ds $tag"; continue; fi
    if [ $tag = vor_cr ]; then args="--passes 5 --save-every -1 --diag-exact"
    else args="--passes 5 --rule lloyd --update seq --save-every -1 --diag-exact"; fi
    echo "--- refine $ds $tag s42 $args $(date +%T)"
    $PIN $PY refine.py $ds --seed 42 --tag $tag $args 2>&1 | grep --line-buffered -E "^(init|ep |diag|saved|Traceback|.*Error)" | sed -u -E "s/'labels_md5': '[0-9a-f]+'//"
  done
done
echo "VORONOI_NODE1_DONE $(date)"
