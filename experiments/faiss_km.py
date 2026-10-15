#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FAISS k-means 基线（Lloyd，GEMM 分配）。
用法: faiss_km.py <ds> <seed> [--niter 25] [--mpc 256] [--name faiss] [--init hier]
  --mpc 256    FAISS 默认的"每簇最多 256 个训练点"抽样；默认不抽样（全量）。
  --niter      迭代轮数；faiss.Kmeans 的默认是 25，IndexIVF 训练（Level1Quantizer）的默认是 10。
               FAISS 的 IVF 默认配置 = --niter 10 --mpc 256（name 用 faiss_ivf）。
  --init hier  从同种子分层 SuperKMeans 的质心暖启动（warm Lloyd；--niter 1 即"一次精确分配 + 均值更新"）。
               计时 = 分层版的 train_ms + 本脚本的 Lloyd 时间（与精修方法同一口径）。"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, c_of, cpp_train_s, read_fvecs, res_dir, write_fvecs  # noqa: E402

import faiss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds")
ap.add_argument("seed", type=int)
ap.add_argument("--niter", type=int, default=25)
ap.add_argument("--mpc", type=int, default=2_000_000_000)
ap.add_argument("--name", default="faiss")
ap.add_argument("--init", choices=["random", "hier"], default="random")
ap.add_argument("--threads", type=int, default=20)
a = ap.parse_args()
od = res_dir(a.ds)
tag = f"{a.name}_s{a.seed}"
if os.path.exists(os.path.join(od, tag + ".log")) and not os.environ.get("FORCE"):
    print("skip", a.ds, tag)
    sys.exit(0)
X = read_fvecs(DATASETS[a.ds][0])
n, d = X.shape
c = c_of(a.ds)
faiss.omp_set_num_threads(a.threads)
km = faiss.Kmeans(d, c, niter=a.niter, seed=a.seed, verbose=False, max_points_per_centroid=a.mpc)
extra_s = 0.0
init = None
if a.init == "hier":
    init = read_fvecs(os.path.join(od, f"hier_s{a.seed}.centroids.fvecs"))
    extra_s = cpp_train_s(os.path.join(od, f"hier_s{a.seed}.log"))
t0 = time.perf_counter()
km.train(X, init_centroids=init) if init is not None else km.train(X)
t = time.perf_counter() - t0
write_fvecs(os.path.join(od, tag + ".centroids.fvecs"), km.centroids.reshape(c, d))
with open(os.path.join(od, tag + ".log"), "w") as fh:
    fh.write(f"config,faiss_kmeans niter={a.niter} mpc={a.mpc} init={a.init} c={c} seed={a.seed} omp={faiss.omp_get_max_threads()}\n")
    if extra_s:
        fh.write(f"hier_ms,{extra_s * 1000:.1f}\n")
        fh.write(f"lloyd_ms,{t * 1000:.1f}\n")
    fh.write(f"train_ms,{(t + extra_s) * 1000:.1f}\n")
print(f"{a.ds} {tag} train {t + extra_s:.1f}s (lloyd {t:.1f}s)", flush=True)
