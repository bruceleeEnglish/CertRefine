#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FAISS two-level clustering (faiss.contrib.clustering.two_level_clustering): nc1 = round(sqrt(c)) first-level
clusters, then c second-level centroids with rebalancing (second-level centroids allotted by first-level cluster size).
The first level is faiss.Kmeans with 25 iterations and at most 2000 points per centroid (hard-coded in FAISS); each
second-level clustering is faiss.Kmeans with its defaults (25 iterations, at most 256 points per centroid).
The CPU counterpart of the hierarchical builds of ScaNN/SPANN/cuVS that the reviews named.
Writes <RES>/<ds>/faiss_2lc_s<seed>.centroids.fvecs and .log (train_ms).
Usage: faiss_2lc.py <ds> <seed> [--threads 20]"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, c_of, read_fvecs, res_dir, write_fvecs  # noqa: E402

import faiss  # noqa: E402
from faiss.contrib.clustering import two_level_clustering  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds")
ap.add_argument("seed", type=int)
ap.add_argument("--threads", type=int, default=20)
ap.add_argument("--name", default="faiss_2lc")
a = ap.parse_args()
od = res_dir(a.ds)
tag = f"{a.name}_s{a.seed}"
if os.path.exists(os.path.join(od, tag + ".log")) and not os.environ.get("FORCE"):
    print("skip", a.ds, tag)
    sys.exit(0)
faiss.omp_set_num_threads(a.threads)
X = read_fvecs(DATASETS[a.ds][0])
n, d = X.shape
c = c_of(a.ds)
nc1 = int(round(np.sqrt(c)))
t0 = time.perf_counter()
cent, _ = two_level_clustering(X, nc1, c, rebalance=True, clustering_niter=25, seed=a.seed, verbose=False)
t = time.perf_counter() - t0
cent = np.ascontiguousarray(cent, dtype=np.float32)
assert cent.shape == (c, d), cent.shape
write_fvecs(os.path.join(od, tag + ".centroids.fvecs"), cent)
with open(os.path.join(od, tag + ".log"), "w") as fh:
    fh.write(f"config,faiss two_level_clustering nc1={nc1} nc2={c} rebalance=1 clustering_niter=25 kmeans_defaults seed={a.seed} omp={a.threads}\n")
    fh.write(f"train_ms,{t * 1000:.1f}\n")
print(f"{a.ds} {tag} train {t:.1f}s (nc1={nc1})", flush=True)
