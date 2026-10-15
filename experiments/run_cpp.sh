#!/usr/bin/env bash
# C++ 基线（SuperKMeans / SuperCDKM 引擎）。每个配置一次训练，日志含 train_ms（只计 Train()）。
# 用法: run_cpp.sh <ds> <seed> <cfg...>
#   hier         分层 SuperKMeans（√c 个 meso 簇 3 轮 + 簇内 5 轮），并导出分层版自身的点分配（*.labels.u32）
#   hor          分层 + SuperKMeans 自带的 5 轮 refinement
#   hsl_cpp      以分层质心初始化，引擎内 shortlist Lloyd 5 轮（第 0 轮为全量 GEMM 分配，其余 4 轮只比 K=64 个候选）
#   skm_default  SuperKMeans 出厂配置（抽样 30%、每簇 ≤256 点、recall 早停，≤10 轮）。
#                早停用的 1000 条查询从训练数据中抽样（sample_queries），不使用评测查询。
#                （2026-09-24 之前的版本把评测查询传给了早停，属于测试集泄漏；旧结果移到 _superseded/。）
#   hor1, hor2   分层 + SuperKMeans 自带 refinement 1 / 2 轮（与 hor 的 5 轮构成同一方法的多个工作点）
#   lloyd        SuperKMeans 全量 Lloyd 25 轮（ADSampling 维度剪枝）
#   lloyd_exact  同上但关闭剪枝（每轮精确 GEMM；用于排查剪枝的影响）
#   bcdkm        批量 CDKM 引擎：全量 25 轮，Hartigan 判据 + 候选 K=64（每 5 轮刷新）
set -uo pipefail
source "$(dirname "$0")/env.sh" "${NODE:-0}"
DS=$1; SEED=$2; shift 2
OUT=$($PY -c "import sys; sys.path.insert(0,'$REPRO'); from common import res_dir; print(res_dir('$DS'))")
read -r B Q C <<< "$($PY -c "import sys; sys.path.insert(0,'$REPRO'); from common import DATASETS, c_of; b,q,g,_=DATASETS['$DS']; print(b, q, c_of('$DS'))")"
NQ=2000
run() {
  local name=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local tag=${name}_s$SEED
  if [ -f "$OUT/$tag.log" ] && grep -q '^train_ms' "$OUT/$tag.log" && [ -z "${FORCE:-}" ]; then echo "skip $DS $tag"; return; fi
  echo "=== [$DS] $tag  env: ${envs[*]}  args: $*   $(date +%T)"
  env SKM_SEED=$SEED "${envs[@]}" $PIN $SCDKM_BIN $B $C "$1" "$2" "$3" $OUT/$tag "${@:4}" > $OUT/$tag.log 2>&1
  echo "    rc=$? train_s=$(grep '^train_ms' $OUT/$tag.log | cut -d, -f2 | awk '{printf "%.2f", $1/1000}')"
}
for cfg in "$@"; do
  case $cfg in
    hier)        run hier SKM_HIER=1 SKM_SAVE_LABELS=1 -- 10 0 0;;
    hor)         run hor SKM_HIER=1 SKM_HIER_REFINE=5 -- 10 0 0;;
    hor1)        run hor1 SKM_HIER=1 SKM_HIER_REFINE=1 -- 10 0 0;;
    hor2)        run hor2 SKM_HIER=1 SKM_HIER_REFINE=2 -- 10 0 0;;
    hsl_cpp)     run hsl_cpp SKM_INIT=$OUT/hier_s$SEED.centroids.fvecs SKM_CANDK=64 -- 5 0 0;;
    skm_default) run skm_default SKM_SAMPLING=0.3 SKM_MAXPPC=256 SKM_SAMPLE_QUERIES=1000 -- 10 0 1;;
    lloyd)       run lloyd -- 25 0 0;;
    lloyd_exact) run lloyd_exact SKM_BLAS_ONLY=1 -- 25 0 0;;
    bcdkm)       run bcdkm SKM_CANDK=64 -- 25 1 0;;
    *) echo "unknown cfg $cfg";;
  esac
done
