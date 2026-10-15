#!/usr/bin/env bash
# 主链之后的计时实验（NUMA node0，串行）：等 chain_main.sh 结束后依次运行。
# 用法: setsid nohup bash chain_extra.sh [stage...] > $ROOT/logs/extra.log 2>&1 < /dev/null &
#   stage: retime single labels voronoi cpp msmarco maint qps deep_flat   （默认全部，按此顺序）
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 0 >/dev/null
DS8="sift glove200 yahoo gist mxbai dbpedia wiki deep10m"
until grep -q "^MAIN_DONE" $ROOT/logs/main.log 2>/dev/null; do sleep 120; done
echo "main chain done, starting extra $(date)"
ref() {
  local ds=$1 s=$2 tag=$3; shift 3
  local dir; dir=$($PY -c "import sys; sys.path.insert(0,'.'); from common import res_dir; print(res_dir('$ds'))")
  if [ -f $dir/${tag}_s$s.refine.json ] && [ -z "${FORCE:-}" ]; then echo "skip $ds $tag s$s"; return; fi
  echo "--- refine $ds $tag s$s $* $(date +%T)"
  $PIN $PY refine.py $ds --seed $s --tag $tag "$@" 2>&1 | grep --line-buffered -E "^(init|ep |saved|Traceback|.*Error)" | sed -u -E "s/'labels_md5': '[0-9a-f]+'//"
}
stage_retime() {   # 重跑计时离群的运行（起步分配时间远高于同数据集其它运行；结果确定，质心不变，只更新计时）
  FORCE=1 ref glove200 1 hslc --passes 5 --rule lloyd --update batch --cert 1 --save-every 1
  # 计时离群的运行（名单格式 数据集:方法[:种子]，种子缺省 42）：重跑并删除旧评测，让评测进程重新评测
  for item in $(cat $ROOT/logs/retime_list.txt 2>/dev/null); do
    IFS=: read -r ds cfg sd <<< "$item"; sd=${sd:-42}
    echo "--- retime $ds $cfg s$sd $(date +%T)"
    rm -f $RES/$ds/${cfg}_s$sd.scan.json $RES/$ds/${cfg}_s$sd.hits.npy
    case $cfg in faiss) FORCE=1 $PIN $PY faiss_km.py $ds $sd;; faiss_sampled) FORCE=1 $PIN $PY faiss_km.py $ds $sd --mpc 256 --name faiss_sampled;;
                 *) FORCE=1 bash run_cpp.sh $ds $sd $cfg;; esac
  done
}
stage_single() {
  for ds in $DS8; do
    [ -f $RES/$ds/single_s42.refine.json ] && continue
    echo "--- single-bound $ds $(date +%T)"
    $PIN $PY refine_single.py $ds 2>&1 | grep --line-buffered -E "^ep|Error"
  done
}
stage_labels() {   # CertRefine-L：直接从分层标签起步（无精确分配），补齐 seed 1、2
  for ds in $DS8; do for s in 1 2; do ref $ds $s cr_labels --passes 5 --init labels --save-every 0; done; done
  # 孪生对照：同样从分层标签起步的批量 Lloyd + 证书（检验起步远离 Voronoi 分配时 Hartigan 顺序更新的作用）
  for ds in $DS8; do ref $ds 42 hslc_labels --passes 5 --rule lloyd --update batch --cert 1 --init labels --save-every 0; done
  # 更新规则/判据的消融补齐 3 个种子（seed 42 已在 core 中）：顺序 Lloyd、批量 Hartigan
  for ds in $DS8; do for s in 1 2; do
    ref $ds $s sl --passes 5 --rule lloyd --update seq --cert 1 --save-every 0
    ref $ds $s hb --passes 5 --rule hartigan --update batch --cert 1 --save-every 0
  done; done
}
stage_voronoi() {  # 诊断：精修结束时与 Voronoi 单元的一致率（Hartigan vs 顺序 Lloyd，seed 42，不保存质心）
  for ds in $DS8; do
    ref $ds 42 vor_cr --passes 5 --save-every -1 --diag-exact
    ref $ds 42 vor_sl --passes 5 --rule lloyd --update seq --save-every -1 --diag-exact
  done
}
stage_cpp() {      # C++ 移植版交叉验证（seed 42）：与 numba 原型对照召回、跳过率、逐轮分配一致性与耗时
  for ds in $DS8; do
    local dir; dir=$RES/$ds
    [ -f $dir/xcr_s42.refine.json ] && continue
    read -r B C <<< "$($PY -c "import sys; sys.path.insert(0,'.'); from common import DATASETS, c_of; print(DATASETS['$ds'][0], c_of('$ds'))")"
    echo "--- cpp certrefine $ds $(date +%T)"
    env CR_CERT=1 CR_SAVE=ends $PIN $ROOT/SuperCDKM/build/examples/certrefine.out $B $dir/hier_s42 $C $dir/xcr_s42 2>&1 | tail -2
    env CR_CERT=0 CR_SAVE=none $PIN $ROOT/SuperCDKM/build/examples/certrefine.out $B $dir/hier_s42 $C $dir/xcr_nocert_s42 2>&1 | tail -1
  done
}
stage_maint() {
  for ds in sift yahoo; do for w in uniform clustered drift; do
    [ -f $RES/$ds/maint/${w}_s42.json ] && continue
    echo "--- maint $ds $w $(date +%T)"
    $PIN $PY maint.py $ds --workload $w 2>&1 | grep --line-buffered -vE "^$"
  done; done
}
stage_msmarco() {
  until [ -f $DATA_MSMARCO/msmarco_meta.json ]; do sleep 120; done
  echo "########## msmarco $(date)"
  for s in 42 1 2; do
    bash run_cpp.sh msmarco $s hier
    ref msmarco $s cr --passes 5 --save-every 0
    ref msmarco $s cr_labels --passes 5 --init labels --save-every 0
  done
  ref msmarco 42 hslc --passes 5 --rule lloyd --update batch --cert 1 --save-every 0
  ref msmarco 42 sl --passes 5 --rule lloyd --update seq --cert 1 --save-every 0
  bash run_cpp.sh msmarco 42 hsl_cpp hor
  for s in 42 1 2; do bash run_cpp.sh msmarco $s skm_default; done
  $PIN $PY faiss_km.py msmarco 42 --mpc 256 --name faiss_sampled
  $PIN $PY faiss_km.py msmarco 42
  touch $RES/msmarco/.train_done
}
stage_qps() {
  for ds in $DS8; do
    echo "--- qps $ds $(date +%T)"
    $PIN $PY eval_qps.py $ds hier_s42 cr_s42_ep1 cr_s42_ep5 cr_labels_s42_ep5 hslc_s42_ep5 hsl_cpp_s42 hor_s42 skm_default_s42 lloyd_s42 faiss_s42 2>&1 | grep --line-buffered -vE "^$"
  done
}
stage_deep_flat() {
  for s in 1 2; do
    $PIN $PY faiss_km.py deep10m $s
    bash run_cpp.sh deep10m $s lloyd skm_default
  done
}
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(retime single labels voronoi cpp msmarco maint qps deep_flat)
for st in "${STAGES[@]}"; do stage_$st; done
echo "EXTRA_DONE ${STAGES[*]} $(date)"
