# 实验的服务器环境（2×Xeon Gold 5218R，每路 20 物理核，2 个 NUMA 节点，125 GB 内存，2×RTX 3090）
# 用法：source env.sh [node]   node 默认 0；所有计时实验只在 node0 上串行跑，评测/编码放 node1。
export ROOT=/path/to/workspace
export REPRO=$ROOT/experiments
export SCDKM_DATA=/path/to/data
export DATA10M=$ROOT/data10m
export DATA_MSMARCO=$ROOT/data_msmarco
export RES=${RES:-$REPRO/results}
export SCDKM_BIN=$ROOT/SuperCDKM/build/examples/scdkm_exp_bench.out
export PY=/path/to/conda/envs/supercdkm_repro/bin/python   # numpy/numba/faiss(CPU)
export PY_ENC=$ROOT/venv_enc/bin/python                                        # torch(cu128) + sentence-transformers，仅用于 MS MARCO 编码
export NODE=${1:-${NODE:-0}}
export CPUS=$(cat /sys/devices/system/node/node$NODE/cpulist)
export PIN="taskset -c $CPUS"
# 20 个工作线程 = 一个 NUMA 节点的物理核数（与原论文所有计时一致）
export OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20 NUMBA_NUM_THREADS=20
mkdir -p $RES
