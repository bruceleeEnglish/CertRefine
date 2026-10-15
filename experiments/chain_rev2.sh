#!/usr/bin/env bash
# 第二轮补充实验（2026-09-26；NUMA node0 串行，计时有效）。只补种子、补基线工作点、加对照；不改算法，
# 不重跑任何已有五种子的主表配置。每个阶段都可重入：已有结果文件的运行自动跳过。
#   msmarco5   MS MARCO 种子 3/4，16 个标签（故意跳过 faiss_sampled：补充材料行保持三种子）          ≈3.5 h
#   nocert     顺序 Hartigan 无证书运行补种子 1–4（开/关证书分区一致性 8/8 → 40/40）                  ≈0.3 h
#   sens       敏感性表补种子 1/2：K 扫描（7 个 1M 集）、snap 起步、标签起步一致率（只算诊断）、单界    ≈1 h
#   cscale2    SIFT c 扫描补种子 1/2 与缺失基线（hor1、faiss_ivf、faiss；s42 补 hor1、faiss_ivf）      ≈1.5 h
#   maint2     维护实验补种子 1/2（sift、yahoo × uniform/clustered/drift）                            ≈2 h
#   fsweep     FAISS 迭代 × 采样扫描（seed 42，8 集）：等召回下"按数据集调优的 FAISS"                  ≈1.5 h
#   control    维护"无插入、同样轮数"对照（seed 42；分离修复收益与原划分的继续收敛）                   ≈0.5 h
#   starts     近似起步 greedy / ivfgemm / ann（seed 42，8 集）                                       ≈0.3 h
#   minibatch  scikit-learn MiniBatchKMeans 基线（8 集 × 3 种子）                                    ≈2–3 h
#   k2style    候选图刷新 R=1 的 k²-means 式配置：先在 SIFT 冒烟测试，通过后 8 集 × 3 种子              ≈0.5 h
# 用法: setsid nohup bash chain_rev2.sh [stage...] > $ROOT/logs/rev2.log 2>&1 < /dev/null &
# 前提：两个 GPU 评测守护进程在跑（chain_eval_gpu.sh 评 8 集与 sift_c*；chain_msmarco_eval_gpu.sh 评 MS MARCO）。
#       msmarco5 开始前删 $RES/msmarco/.rev_train_done（守护进程的退出标记），结束后由本脚本重新 touch。
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 0 >/dev/null
DS8="sift glove200 yahoo gist mxbai dbpedia wiki deep10m"
DS7="sift glove200 yahoo gist mxbai dbpedia wiki"
ref() {
  local ds=$1 s=$2 tag=$3; shift 3
  local dir; dir=$($PY -c "import sys; sys.path.insert(0,'.'); from common import res_dir; print(res_dir('$ds'))")
  if [ -f $dir/${tag}_s$s.refine.json ] && [ -z "${FORCE:-}" ]; then echo "skip $ds $tag s$s"; return; fi
  echo "--- refine $ds $tag s$s $* $(date +%T)"
  $PIN $PY refine.py $ds --seed $s --tag $tag "$@" 2>&1 | grep --line-buffered -E "^(init|saved|diag|Traceback|.*Error)"
}
fk() {  # faiss_km.py 的包装：<ds> <seed> <name> [args]（脚本自身在 <name>_s<seed>.log 存在时跳过）
  local ds=$1 s=$2 name=$3; shift 3
  $PIN $PY faiss_km.py $ds $s --name $name "$@" 2>&1 | grep --line-buffered -vE "^$"
}
new_baselines() {  # <ds> <seed>：与 chain_rev.sh 相同的一组基线
  local ds=$1 s=$2
  bash run_cpp.sh $ds $s skm_default hor1 hor2
  fk $ds $s faiss_ivf --niter 10 --mpc 256
  for k in 1 2 3; do fk $ds $s warm$k --init hier --niter $k; done
}
stage_msmarco5() {
  local ds=msmarco
  echo "########## msmarco5 $(date)"
  rm -f $RES/msmarco/.rev_train_done          # 只影响 MS MARCO 评测守护进程的退出时机；它每 120 s 重扫一次目录
  for s in 3 4; do
    bash run_cpp.sh $ds $s hier               # refine.py 需要 hier_s$s.labels.u32 与 .centroids.fvecs
    ref $ds $s cr          --passes 5 --save-every 0
    ref $ds $s cr_labels   --passes 5 --init labels --save-every 0
    ref $ds $s hslc        --passes 5 --rule lloyd --update batch --cert 1 --save-every 0
    ref $ds $s hslc_labels --passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every 0
    ref $ds $s sl          --passes 5 --rule lloyd --update seq --cert 1 --save-every 0
    bash run_cpp.sh $ds $s hsl_cpp hor
    new_baselines $ds $s                       # skm_default hor1 hor2 faiss_ivf warm1-3
    $PIN $PY faiss_km.py $ds $s                # FAISS-full（25 轮全量），约 40 min
    # faiss_sampled 故意不跑（补充材料行保持三种子，正文图注写明）
  done
  touch $RES/msmarco/.rev_train_done
}
stage_nocert() {
  echo "########## nocert $(date)"
  for ds in $DS8; do for s in 1 2 3 4; do
    ref $ds $s cr_nocert --passes 5 --cert 0 --save-every -1
  done; done
}
stage_sens() {
  echo "########## sens $(date)"
  for ds in $DS7; do for s in 1 2; do for K in 16 32 128 256; do
    ref $ds $s cr_K$K --K $K --passes 5 --save-every 0
  done; done; done
  for ds in $DS8; do for s in 1 2; do
    ref $ds $s cr_snap  --init snap   --passes 5 --diag-exact --save-every 0
    ref $ds $s cr_labdg --init labels --passes 0 --diag-exact --save-every -1   # 只为一致率列（诊断不计时）
    if [ ! -f $RES/$ds/single_s$s.refine.json ]; then
      echo "--- single-bound $ds s$s $(date +%T)"
      $PIN $PY refine_single.py $ds --seed $s 2>&1 | grep --line-buffered -E "^ep|Error"
    fi
  done; done
}
stage_cscale2() {
  echo "########## cscale2 $(date)"
  for C in 1000 2000 8000; do
    export SCDKM_C=$C
    for s in 1 2; do
      bash run_cpp.sh sift $s hier hsl_cpp hor hor1 skm_default
      ref sift $s cr          --passes 5 --save-every 0
      ref sift $s cr_labels   --passes 5 --init labels --save-every 0
      ref sift $s hslc        --passes 5 --rule lloyd --update batch --cert 1 --save-every 0
      ref sift $s hslc_labels --passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every 0
      fk sift $s faiss_ivf --niter 10 --mpc 256
      $PIN $PY faiss_km.py sift $s
    done
    bash run_cpp.sh sift 42 hor1
    fk sift 42 faiss_ivf --niter 10 --mpc 256
    unset SCDKM_C
  done
}
stage_maint2() {
  echo "########## maint2 $(date)"
  for ds in sift yahoo; do for w in uniform clustered drift; do for s in 1 2; do
    [ -f $RES/$ds/maint/${w}_s$s.json ] && { echo "skip maint $ds $w s$s"; continue; }
    echo "--- maint $ds $w s$s $(date +%T)"
    $PIN $PY maint.py $ds --workload $w --seed $s 2>&1 | grep --line-buffered -vE "^$"
  done; done; done
}
stage_fsweep() {
  echo "########## fsweep $(date)"
  for ds in $DS8; do
    for it in 5 10 25; do for m in 39 64 128; do
      fk $ds 42 faiss_it${it}_m${m} --niter $it --mpc $m
    done; done
    for it in 5 10; do fk $ds 42 faiss_it${it}_full --niter $it; done   # 25/full = faiss，10/256 = faiss_ivf，已有
  done
}
stage_control() {
  echo "########## control $(date)"
  for ds in sift yahoo; do for w in uniform clustered drift; do
    [ -f $RES/$ds/maint/${w}_s42_control.json ] && { echo "skip control $ds $w"; continue; }
    echo "--- control $ds $w $(date +%T)"
    $PIN $PY maint.py $ds --workload $w --seed 42 --control 2>&1 | grep --line-buffered -vE "^$"
  done; done
}
stage_starts() {
  echo "########## starts $(date)"
  for ds in $DS8; do for init in greedy ivfgemm ann; do
    ref $ds 42 cr_$init --init $init --passes 5 --diag-exact --save-every 0
  done; done
}
stage_minibatch() {
  echo "########## minibatch $(date)"
  for ds in $DS8; do for s in 42 1 2; do
    $PIN $PY sk_minibatch.py $ds $s 2>&1 | grep --line-buffered -vE "^$"
  done; done
}
stage_k2style() {
  echo "########## k2style $(date)"
  ref sift 42 k2smoke --passes 5 --rule lloyd --update batch --cert 1 --init labels --refresh 1 --save-every -1
  $PY k2smoke_check.py && touch $RES/sift/.k2smoke_ok
  if [ -f $RES/sift/.k2smoke_ok ]; then
    for ds in $DS8; do for s in 42 1 2; do
      ref $ds $s k2style --passes 5 --rule lloyd --update batch --cert 1 --init labels --refresh 1 --save-every 0
    done; done
  else
    echo "k2style skipped: smoke test failed"
  fi
}
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(msmarco5 nocert sens cscale2 maint2 fsweep control starts minibatch k2style)
for st in "${STAGES[@]}"; do stage_$st; echo "STAGE_DONE $st $(date)"; done
echo "REV2_DONE ${STAGES[*]} $(date)"
