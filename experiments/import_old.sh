#!/usr/bin/env bash
# 把上一轮（2026-09-22，同一台机器、同一 C++ 二进制与同一 FAISS 版本、单 NUMA 节点 20 线程）已跑好的
# 与分层初始化无关的平坦基线（seed 42 及已有的 SKM default 三种子）导入本轮结果目录并改成新命名。
# 这些方法的训练过程与本轮完全相同，只重新评测（eval_scan.py 会对导入的质心重新做等扫描评测）。
set -uo pipefail
source "$(dirname "$0")/env.sh" 0 >/dev/null
OLD=$ROOT/results
cp_pair() { # <ds> <oldname> <newname>
  local ds=$1 o=$2 nw=$3
  [ -f $OLD/$ds/$o.centroids.fvecs ] || return 0
  mkdir -p $RES/$ds
  if [ ! -f $RES/$ds/$nw.centroids.fvecs ]; then
    cp $OLD/$ds/$o.centroids.fvecs $RES/$ds/$nw.centroids.fvecs
    cp $OLD/$ds/$o.log $RES/$ds/$nw.log
    echo "imported $ds/$o -> $nw"
  fi
}
for ds in sift glove200 yahoo gist mxbai dbpedia wiki deep10m; do
  cp_pair $ds faiss faiss_s42
  cp_pair $ds faiss_sampled faiss_sampled_s42
  cp_pair $ds lloyd_full lloyd_s42
  cp_pair $ds cand64_full bcdkm_s42
  cp_pair $ds cand64_samp bcdkm_samp_s42
  cp_pair $ds skm_default skm_default_s42
  cp_pair $ds skm_default_s1 skm_default_s1
  cp_pair $ds skm_default_s2 skm_default_s2
  cp_pair $ds sk_elkan sk_elkan_s42
  cp_pair $ds sk_lloyd sk_lloyd_s42
  cp_pair $ds sk_minibatch sk_minibatch_s42
done
echo "# provenance: imported from $OLD on $(date)" > $RES/IMPORTED_FROM_2026-09-22.txt
