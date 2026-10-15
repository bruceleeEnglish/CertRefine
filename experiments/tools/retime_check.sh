#!/usr/bin/env bash
# Timing reproducibility check for the runs flagged by tools/timing_outliers.py.
# Each listed run is trained again with the same seed, under the same protocol (node0, 20 threads, pinned, one
# run at a time), but into a scratch result directory, so the reported results are never overwritten.
# The script prints the original and the repeated training time. A deviation that reproduces is seed-dependent
# work (pruning, early termination, convergence) and the run is kept as measured; a deviation that does not
# reproduce is interference, and the run is then re-timed in place with the retime stage of chain_extra.sh.
# Usage (server):  bash tools/retime_check.sh <list>     list lines: <collection>:<method>:<seed>
set -uo pipefail
cd "$(dirname "$0")/.."
LIST=$1
source ./env.sh 0 >/dev/null
MAIN=$RES
export RES=$ROOT/results_retime_check
mkdir -p $RES
for item in $(grep -v '^#' "$LIST" | cut -d' ' -f1); do
  IFS=: read -r ds cfg sd <<< "$item"
  mkdir -p $RES/$ds
  # methods that start from the hierarchical centroids of the same seed read them from the scratch directory
  ln -sf $MAIN/$ds/hier_s$sd.centroids.fvecs $RES/$ds/hier_s$sd.centroids.fvecs 2>/dev/null
  ln -sf $MAIN/$ds/hier_s$sd.log $RES/$ds/hier_s$sd.log 2>/dev/null
  case $cfg in
    faiss)          FORCE=1 $PIN $PY faiss_km.py $ds $sd >/dev/null 2>&1;;
    faiss_ivf)      FORCE=1 $PIN $PY faiss_km.py $ds $sd --niter 10 --mpc 256 --name faiss_ivf >/dev/null 2>&1;;
    faiss_sampled)  FORCE=1 $PIN $PY faiss_km.py $ds $sd --mpc 256 --name faiss_sampled >/dev/null 2>&1;;
    warm1|warm2|warm3) FORCE=1 $PIN $PY faiss_km.py $ds $sd --init hier --niter ${cfg#warm} --name $cfg >/dev/null 2>&1;;
    hier|hor|hor1|hor2|hsl_cpp|skm_default|lloyd|lloyd_exact|bcdkm) FORCE=1 bash run_cpp.sh $ds $sd $cfg >/dev/null 2>&1;;
    *) echo "$item: not supported (refinement runs report their own phase timings)"; continue;;
  esac
  old=$(grep '^train_ms' $MAIN/$ds/${cfg}_s$sd.log | cut -d, -f2)
  new=$(grep '^train_ms' $RES/$ds/${cfg}_s$sd.log | cut -d, -f2)
  awk -v i="$item" -v o="$old" -v n="$new" 'BEGIN{printf "%-28s original %9.2f s   repeated %9.2f s   ratio %.3f\n", i, o/1000, n/1000, n/o}'
done
echo "RETIME_CHECK_DONE $(date)"
