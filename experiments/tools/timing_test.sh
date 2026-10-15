#!/usr/bin/env bash
# 在 node1 上做 C++ 与 numba 版 CertRefine 的计时对照：先暂停 node1 上的评测进程（SIGSTOP），测完自动恢复（SIGCONT）。
# 用法: bash tools/timing_test.sh <ds> <c>
set -uo pipefail
cd /path/to/workspace
source experiments/env.sh 1 >/dev/null
DS=$1; C=$2
P=$(pgrep -u user -f "[b]in/python eval_scan.py" | tr "\n" " ")
echo "pausing eval: $P"
[ -n "$P" ] && kill -STOP $P
trap '[ -n "$P" ] && kill -CONT $P && echo "resumed eval"' EXIT
sleep 2
T=results_test/$DS; mkdir -p $T; cp -n results/$DS/hier_s42.* $T/ 2>/dev/null
BIN=SuperCDKM/build/examples/certrefine.out
B=$($PY -c "import sys; sys.path.insert(0,'experiments'); from common import DATASETS; print(DATASETS['$DS'][0])")
env CR_CERT=1 CR_SAVE=none $PIN $BIN $B $T/hier_s42 $C $T/xcr_timing_s42 2>&1 | tail -3
$PIN $PY experiments/refine.py $DS --seed 42 --tag tmpnumba --passes 5 --save-every -1 2>&1 | grep -E "^init|epoch.: (1|5)," | cut -c1-200
$PY - <<EOF
import json
a = json.load(open("$T/xcr_timing_s42.refine.json"))
print("cpp  timing", {k: round(v, 3) for k, v in a["timing"].items()}, "passes", [round(h["epoch_s"], 3) for h in a["hist"][1:]], "total", round(a["hist"][-1]["total_s"], 3))
b = json.load(open("results/$DS/tmpnumba_s42.refine.json"))
print("numba timing", {k: round(v, 3) for k, v in b["timing"].items()}, "passes", [round(h["epoch_s"], 3) for h in b["hist"][1:]], "total", round(b["hist"][-1]["total_s"], 3))
EOF
rm -f results/$DS/tmpnumba_s42.refine.json
