#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scikit-learn MiniBatchKMeans baseline. Random initialization (k-means++ with c in the
thousands is itself c passes over the data), batches of 4096 points, at most 25 epochs, scikit-learn's default early
stopping (max_no_improvement=10, reassignment_ratio=0.01). Writes <RES>/<ds>/sk_minibatch_s<seed>.centroids.fvecs and .log
(train_ms = fit time); the GPU evaluation daemon picks the centroids up automatically.
Usage: sk_minibatch.py <ds> <seed> [--max-iter 25] [--batch 4096] [--name sk_minibatch]"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, c_of, read_fvecs, res_dir, write_fvecs  # noqa: E402

from sklearn.cluster import MiniBatchKMeans  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds")
ap.add_argument("seed", type=int)
ap.add_argument("--max-iter", type=int, default=25)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--name", default="sk_minibatch")
a = ap.parse_args()
od = res_dir(a.ds)
tag = f"{a.name}_s{a.seed}"
if os.path.exists(os.path.join(od, tag + ".log")) and not os.environ.get("FORCE"):
    print("skip", a.ds, tag)
    sys.exit(0)
X = read_fvecs(DATASETS[a.ds][0])
c = c_of(a.ds)
km = MiniBatchKMeans(n_clusters=c, init="random", n_init=1, max_iter=a.max_iter, batch_size=a.batch, random_state=a.seed,
                     reassignment_ratio=0.01)
t0 = time.perf_counter()
km.fit(X)
t = time.perf_counter() - t0
write_fvecs(os.path.join(od, tag + ".centroids.fvecs"), km.cluster_centers_.astype(np.float32))
with open(os.path.join(od, tag + ".log"), "w") as fh:
    fh.write(f"config,sklearn MiniBatchKMeans init=random max_iter={a.max_iter} batch={a.batch} c={c} seed={a.seed}\n")
    fh.write(f"n_iter,{km.n_iter_}\nn_steps,{km.n_steps_}\n")
    fh.write(f"train_ms,{t * 1000:.1f}\n")
print(f"{a.ds} {tag} train {t:.1f}s (epochs {km.n_iter_}, steps {km.n_steps_})", flush=True)
