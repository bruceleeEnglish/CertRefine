#!/usr/bin/env bash
# chain_rev2.sh 之后的补充（等 REV2_DONE 出现后自动开始；node0 串行；全部可重入）：
#   maint42    维护实验 seed 42 用带逐查询输出的 maint.py 重跑（旧结果移到 maint/_v1/），使三种子都能做种子×查询自助法
#   lists      "精修后的列表直接使用"与"候选受限最终分配"两种协议变体（crf / crf_labels / bf_labels，含标签文件与计时；
#              8 集 + MS MARCO，种子 42/1/2）：最终精确分配占精修构建 42–67% 的时间
#   fsweep_ms  MS MARCO 上的 FAISS 迭代 × 采样扫描（4 个配置 × 3 种子）
#   fsweep2    FAISS 扫描在 7 个 1M 集补种子 1/2（Deep-10M 只保留 seed 42）
#   twolevel   FAISS 自带的 two_level_clustering（8 集 + MS MARCO × 3 种子）
#   peakmem    峰值内存（GNU time -v）：Deep-10M 与 MS MARCO 上 H-5 / H-L / B-L / FAISS-IVF / HOR，seed 42，写入 results/peakmem/
# 用法: setsid nohup bash chain_rev2b.sh [stage...] > $ROOT/logs/rev2b.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 0 >/dev/null
DS8="sift glove200 yahoo gist mxbai dbpedia wiki deep10m"
DS7="sift glove200 yahoo gist mxbai dbpedia wiki"
until grep -q "^REV2_DONE" $ROOT/logs/rev2.log 2>/dev/null; do sleep 120; done
echo "rev2 done, starting rev2b $(date)"
# MS MARCO 评测守护进程在 .rev_train_done 存在且全部评完后会退出；本链还会产生新的 MS MARCO 质心，所以先删标记并确保它在跑
rm -f $RES/msmarco/.rev_train_done
if ! pgrep -f chain_msmarco_eval_gpu.sh >/dev/null; then
  (setsid nohup bash chain_msmarco_eval_gpu.sh >> $ROOT/logs/msmarco_eval_gpu.log 2>&1 < /dev/null &)
  echo "restarted chain_msmarco_eval_gpu.sh $(date)"
fi
pgrep -f chain_eval_gpu.sh >/dev/null || echo "WARNING: chain_eval_gpu.sh is not running"
ref() {
  local ds=$1 s=$2 tag=$3; shift 3
  local dir; dir=$($PY -c "import sys; sys.path.insert(0, '.'); from common import res_dir; print(res_dir('$ds'))")
  if [ -f $dir/${tag}_s$s.refine.json ] && [ -z "${FORCE:-}" ]; then echo "skip $ds $tag s$s"; return; fi
  echo "--- refine $ds $tag s$s $* $(date +%T)"
  $PIN $PY refine.py $ds --seed $s --tag $tag "$@" 2>&1 | grep --line-buffered -E "^(init|saved|diag|final|Traceback|.*Error)"
}
fk() {
  local ds=$1 s=$2 name=$3; shift 3
  $PIN $PY faiss_km.py $ds $s --name $name "$@" 2>&1 | grep --line-buffered -vE "^$"
}
stage_maint42() {
  echo "########## maint42 $(date)"
  for ds in sift yahoo; do
    mkdir -p $RES/$ds/maint/_v1
    for w in uniform clustered drift; do
      [ -f $RES/$ds/maint/${w}_s42_frozen.pqb.npy ] && { echo "skip maint42 $ds $w"; continue; }
      mv -f $RES/$ds/maint/${w}_s42.json $RES/$ds/maint/${w}_s42_build_hier.log $RES/$ds/maint/${w}_s42_rebuild_hier.log $RES/$ds/maint/_v1/ 2>/dev/null
      echo "--- maint $ds $w s42 (per-query) $(date +%T)"
      $PIN $PY maint.py $ds --workload $w --seed 42 2>&1 | grep --line-buffered -vE "^$"
    done
  done
}
stage_lists() {
  echo "########## lists $(date)"
  for ds in $DS8 msmarco; do for s in 42 1 2; do
    ref $ds $s crf        --passes 5 --save-every 0 --save-labels --final snap
    ref $ds $s crf_labels --passes 5 --init labels --save-every 0 --save-labels --final snap
    ref $ds $s bf_labels  --passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every 0 --save-labels --final snap
  done; done
}
stage_fsweep_ms() {
  echo "########## fsweep_ms $(date)"
  for s in 42 1 2; do
    fk msmarco $s faiss_it5_m64   --niter 5  --mpc 64
    fk msmarco $s faiss_it10_m64  --niter 10 --mpc 64
    fk msmarco $s faiss_it5_m128  --niter 5  --mpc 128
    fk msmarco $s faiss_it10_m128 --niter 10 --mpc 128
  done
}
stage_fsweep2() {
  echo "########## fsweep2 $(date)"
  for ds in $DS7; do for s in 1 2; do
    for it in 5 10 25; do for m in 39 64 128; do
      fk $ds $s faiss_it${it}_m${m} --niter $it --mpc $m
    done; done
    for it in 5 10; do fk $ds $s faiss_it${it}_full --niter $it; done
  done; done
}
stage_twolevel() {
  echo "########## twolevel $(date)"
  for ds in $DS8 msmarco; do for s in 42 1 2; do
    $PIN $PY faiss_2lc.py $ds $s 2>&1 | grep --line-buffered -vE "^$"
  done; done
}
stage_peakmem() {
  echo "########## peakmem $(date)"
  local MAIN=$RES PM=$ROOT/results_peakmem
  mkdir -p $PM $MAIN/peakmem
  for ds in deep10m msmarco; do
    mkdir -p $PM/$ds
    for f in centroids.fvecs labels.u32 log; do ln -sf $MAIN/$ds/hier_s42.$f $PM/$ds/hier_s42.$f; done
    for spec in "cr:--passes 5 --save-every -1" "cr_labels:--passes 5 --init labels --save-every -1" \
                "hslc_labels:--passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every -1"; do
      local tag=${spec%%:*} args=${spec#*:}
      [ -f $MAIN/peakmem/${ds}_$tag.txt ] && continue
      echo "--- peakmem $ds $tag $(date +%T)"
      RES=$PM $PIN /usr/bin/time -v $PY refine.py $ds --seed 42 --tag $tag $args > /dev/null 2> $MAIN/peakmem/${ds}_$tag.txt
    done
    if [ ! -f $MAIN/peakmem/${ds}_faiss_ivf.txt ]; then
      echo "--- peakmem $ds faiss_ivf $(date +%T)"
      RES=$PM $PIN /usr/bin/time -v $PY faiss_km.py $ds 42 --name faiss_ivf --niter 10 --mpc 256 > /dev/null 2> $MAIN/peakmem/${ds}_faiss_ivf.txt
    fi
    if [ ! -f $MAIN/peakmem/${ds}_hor.txt ]; then
      echo "--- peakmem $ds hor $(date +%T)"
      RES=$PM $PIN /usr/bin/time -v bash run_cpp.sh $ds 42 hor > /dev/null 2> $MAIN/peakmem/${ds}_hor.txt
    fi
    grep -H "Maximum resident" $MAIN/peakmem/${ds}_*.txt
  done
}
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(maint42 lists fsweep_ms fsweep2 twolevel peakmem)
for st in "${STAGES[@]}"; do stage_$st; echo "STAGE_DONE $st $(date)"; done
touch $RES/msmarco/.rev_train_done
echo "REV2B_DONE ${STAGES[*]} $(date)"
