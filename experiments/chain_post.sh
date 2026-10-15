#!/usr/bin/env bash
# chain_extra.sh 之后的补充计时实验（NUMA node0，串行）：等 EXTRA_DONE 后运行，避免与其他计时实验重叠。
#   cscale_crl：SIFT 的 c 扫描补上 CertRefine-L（从分层标签起步，c=4000 已在 labels 阶段跑过）
# 用法: setsid nohup bash chain_post.sh > $ROOT/logs/post.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh 0 >/dev/null
until grep -q "^EXTRA_DONE" $ROOT/logs/extra.log 2>/dev/null; do sleep 60; done
echo "extra chain done, starting post $(date)"
for C in 1000 2000 8000; do
  export SCDKM_C=$C
  dir=$($PY -c "import sys; sys.path.insert(0,'.'); from common import res_dir; print(res_dir('sift'))")
  if [ -f $dir/cr_labels_s42.refine.json ]; then echo "skip sift c=$C cr_labels"; unset SCDKM_C; continue; fi
  echo "--- refine sift c=$C cr_labels s42 $(date +%T)"
  $PIN $PY refine.py sift --seed 42 --tag cr_labels --passes 5 --init labels --save-every 0 2>&1 | grep --line-buffered -E "^(init|ep |saved|Traceback|.*Error)" | sed -u -E "s/'labels_md5': '[0-9a-f]+'//"
  unset SCDKM_C
done
echo "POST_DONE $(date)"
