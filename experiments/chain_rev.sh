#!/usr/bin/env bash
# 补充实验（2026-09-24，NUMA node0，计时实验串行）。按优先级分阶段：
#   prep      把有测试集泄漏的旧 SKM-default 结果移到 _superseded/（早停用了评测查询）
#   fix       8 个数据集 × 种子 42/1/2：SKM-default（早停查询改为从训练数据抽样）、FAISS 的 IVF 默认配置（10 轮、每簇 ≤256 点）、
#             HOR 1/2 轮、从分层质心暖启动的 Lloyd 1/2/3 轮（warm1 即"一次精确分配 + 均值更新"）、HSL+cert 从标签起步补种子 1/2
#   assign    最终列表分配的耗时（端到端构建时间 = 训练 + 列表分配）
#   cextra    SIFT c 扫描补 HSL+cert 从标签起步（与 CertRefine-L 对照）
#   sys       线程扩展（1–20 线程）与 batch 大小 B 的扫描（seed 42）
#   msmarco   MS MARCO 所有方法补齐 3 个种子，并加入新基线
#   seeds     8 个数据集补种子 3、4（全部方法），用于把训练随机性纳入统计推断
#   qps       新基线的查询吞吐
# 用法: setsid nohup bash chain_rev.sh [stage...] > $ROOT/logs/rev.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 0 >/dev/null
DS8="sift glove200 yahoo gist mxbai dbpedia wiki deep10m"
ref() {
  local ds=$1 s=$2 tag=$3; shift 3
  local dir; dir=$($PY -c "import sys; sys.path.insert(0,'.'); from common import res_dir; print(res_dir('$ds'))")
  if [ -f $dir/${tag}_s$s.refine.json ] && [ -z "${FORCE:-}" ]; then echo "skip $ds $tag s$s"; return; fi
  echo "--- refine $ds $tag s$s $* $(date +%T)"
  $PIN $PY refine.py $ds --seed $s --tag $tag "$@" 2>&1 | grep --line-buffered -E "^(init|saved|Traceback|.*Error)"
}
fk() {  # faiss_km.py 的包装：<ds> <seed> <name> [args]
  local ds=$1 s=$2 name=$3; shift 3
  $PIN $PY faiss_km.py $ds $s --name $name "$@" 2>&1 | grep --line-buffered -vE "^$"
}
new_baselines() {  # <ds> <seed>
  local ds=$1 s=$2
  bash run_cpp.sh $ds $s skm_default hor1 hor2
  fk $ds $s faiss_ivf --niter 10 --mpc 256
  for k in 1 2 3; do fk $ds $s warm$k --init hier --niter $k; done
}
stage_prep() {
  for d in $RES/*/; do
    [ -d $d ] || continue
    ls $d/skm_default_s* >/dev/null 2>&1 || continue
    mkdir -p $d/_superseded
    mv $d/skm_default_s* $d/_superseded/ 2>/dev/null
    echo "moved leaky skm_default results of $(basename $d)"
  done
}
stage_fix() {
  for ds in $DS8; do
    echo "########## fix $ds $(date)"
    for s in 42 1 2; do new_baselines $ds $s; done
    for s in 1 2; do ref $ds $s hslc_labels --passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every 0; done
  done
}
stage_assign() {
  echo "########## assign $(date)"
  $PIN $PY time_assign.py $DS8 msmarco
}
stage_cextra() {
  for C in 1000 2000 8000; do
    export SCDKM_C=$C
    ref sift 42 hslc_labels --passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every 0
    bash run_cpp.sh sift 42 skm_default   # 旧的 c 扫描 SKM-default 同样有泄漏，已被 prep 移走
    unset SCDKM_C
  done
}
stage_sys() {
  echo "########## sys $(date)"
  for ds in sift gist; do
    for t in 1 2 5 10; do
      NUMBA_NUM_THREADS=$t OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t ref $ds 42 cr_thr$t --passes 5 --threads $t --save-every -1
    done
  done
  for ds in sift glove200; do
    for T in 256 1024 16384; do ref $ds 42 cr_B$T --passes 5 --T $T --save-every 0; done
  done
}
stage_msmarco() {
  echo "########## msmarco-rev $(date)"
  local ds=msmarco
  for s in 42 1 2; do
    ref $ds $s hslc_labels --passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every 0
    new_baselines $ds $s
  done
  for s in 1 2; do
    ref $ds $s sl --passes 5 --rule lloyd --update seq --cert 1 --save-every 0
    ref $ds $s hslc --passes 5 --rule lloyd --update batch --cert 1 --save-every 0
    bash run_cpp.sh $ds $s hsl_cpp hor
    $PIN $PY faiss_km.py $ds $s --mpc 256 --name faiss_sampled
  done
  for s in 1 2; do $PIN $PY faiss_km.py $ds $s; done
  touch $RES/msmarco/.rev_train_done
}
stage_seeds() {
  for ds in $DS8; do
    echo "########## seeds $ds $(date)"
    for s in 3 4; do
      bash run_cpp.sh $ds $s hier
      ref $ds $s cr          --passes 5 --save-every 0
      ref $ds $s cr_labels   --passes 5 --init labels --save-every 0
      ref $ds $s hslc        --passes 5 --rule lloyd --update batch --cert 1 --save-every 0
      ref $ds $s hsl         --passes 5 --rule lloyd --update batch --cert 0 --save-every 0
      ref $ds $s hslc_labels --passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every 0
      ref $ds $s sl          --passes 5 --rule lloyd --update seq --cert 1 --save-every 0
      ref $ds $s hb          --passes 5 --rule hartigan --update batch --cert 1 --save-every 0
      bash run_cpp.sh $ds $s hsl_cpp hor lloyd
      new_baselines $ds $s
      $PIN $PY faiss_km.py $ds $s
    done
  done
}
stage_qps() {
  for ds in $DS8; do
    echo "--- qps-rev $ds $(date +%T)"
    $PIN $PY eval_qps.py $ds skm_default_s42 faiss_ivf_s42 2>&1 | grep --line-buffered -vE "^$"
  done
}
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(prep fix assign cextra sys msmarco seeds qps)
for st in "${STAGES[@]}"; do stage_$st; done
echo "REV_DONE ${STAGES[*]} $(date)"
