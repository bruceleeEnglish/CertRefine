#!/usr/bin/env bash
# 主实验链（计时实验，全部串行跑在 NUMA node0，20 线程）。可重复执行：已完成的步骤会跳过。
# 用法: setsid nohup bash chain_main.sh [stage...] > $ROOT/logs/main.log 2>&1 < /dev/null &
#   stage: core flat cscale ksweep deep_flat   （默认 core flat cscale ksweep）
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 0 >/dev/null
DS7="sift glove200 yahoo gist mxbai dbpedia wiki"
DS8="$DS7 deep10m"
ref() { # <ds> <seed> <tag> [refine.py args...]  已有结果则跳过
  local ds=$1 s=$2 tag=$3; shift 3
  local dir; dir=$($PY -c "import sys; sys.path.insert(0,'.'); from common import res_dir; print(res_dir('$ds'))")
  if [ -f $dir/${tag}_s$s.refine.json ] && [ -z "${FORCE:-}" ]; then echo "skip $ds $tag s$s"; return; fi
  echo "--- refine $ds $tag s$s $* $(date +%T)"
  $PIN $PY refine.py $ds --seed $s --tag $tag "$@" 2>&1 | grep -E "^(init|ep |saved|converged|Traceback|.*Error)" | sed -E "s/'labels_md5': '[0-9a-f]+'//; s/'graph_rebuilt': (True|False), //"
}
stage_core() {
  for ds in $DS8; do
    echo "########## core $ds $(date)"
    for s in 42 1 2; do
      bash run_cpp.sh $ds $s hier
      ref $ds $s cr   --passes 5 --save-every 1
      ref $ds $s hslc --passes 5 --rule lloyd --update batch --cert 1 --save-every 1
      ref $ds $s hsl  --passes 5 --rule lloyd --update batch --cert 0 --save-every 0
      bash run_cpp.sh $ds $s hsl_cpp hor
    done
    ref $ds 42 cr_nocert --passes 5 --cert 0 --save-every -1
    ref $ds 42 sl        --passes 5 --rule lloyd --update seq --cert 1 --save-every 1
    ref $ds 42 hb        --passes 5 --rule hartigan --update batch --cert 1 --save-every 1
    ref $ds 42 cr_labels --passes 5 --init labels --save-every 0 --diag-exact
    ref $ds 42 cr_snap   --passes 5 --init snap --save-every 0 --diag-exact
    touch $RES/$ds/.core_done
  done
}
stage_flat() {
  for ds in $DS7; do
    echo "########## flat $ds $(date)"
    for s in 1 2; do
      $PIN $PY faiss_km.py $ds $s
      bash run_cpp.sh $ds $s lloyd bcdkm skm_default
    done
    bash run_cpp.sh $ds 42 lloyd_exact
    touch $RES/$ds/.flat_done
  done
}
stage_cscale() {
  for C in 1000 2000 8000; do
    echo "########## cscale c=$C $(date)"
    export SCDKM_C=$C
    bash run_cpp.sh sift 42 hier hsl_cpp hor skm_default
    ref sift 42 cr   --passes 5 --save-every 0
    ref sift 42 hslc --passes 5 --rule lloyd --update batch --cert 1 --save-every 0
    $PIN $PY faiss_km.py sift 42
    unset SCDKM_C
  done
}
stage_ksweep() {
  for ds in $DS7; do
    echo "########## ksweep $ds $(date)"
    for K in 16 32 128 256; do ref $ds 42 cr_K$K --K $K --passes 5 --save-every 0; done
  done
}
stage_deep_flat() {
  for s in 1 2; do
    $PIN $PY faiss_km.py deep10m $s
    bash run_cpp.sh deep10m $s lloyd skm_default
  done
  $PIN $PY faiss_km.py deep10m 1 --mpc 256 --name faiss_sampled
  $PIN $PY faiss_km.py deep10m 2 --mpc 256 --name faiss_sampled
}
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(core flat cscale ksweep)
for st in "${STAGES[@]}"; do stage_$st; done
echo "MAIN_DONE ${STAGES[*]} $(date)"
