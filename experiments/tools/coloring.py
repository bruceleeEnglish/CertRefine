#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 Section 4.5 的说法：若要让并发处理的点互不冲突，两个簇 p、q 的候选集 C(p)={p}∪N(p) 与 C(q) 相交时就不能同时处理；
对这个冲突图做贪心着色，统计每种颜色平均包含多少个簇（= 可并行的簇数）。用法: coloring.py [K=64]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import STATIC_DS, read_fvecs, res_dir  # noqa: E402
import refine as R  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
out = {}
for ds in STATIC_DS:
    p = os.path.join(res_dir(ds), "hier_s42.centroids.fvecs")
    if not os.path.exists(p):
        continue
    M0 = read_fvecs(p).astype(np.float64)
    c = M0.shape[0]
    nb = R.build_neigh(M0, K)
    cand = [set([q]) | set(nb[q].tolist()) for q in range(c)]
    # 反向索引：每个簇出现在哪些簇的候选集中
    inv = [[] for _ in range(c)]
    for q in range(c):
        for k in cand[q]:
            inv[k].append(q)
    color = -np.ones(c, np.int64)
    for q in np.argsort([-len(x) for x in cand]):   # 度数大的先着色
        used = set()
        for k in cand[q]:
            for r in inv[k]:
                if color[r] >= 0:
                    used.add(int(color[r]))
        col = 0
        while col in used:
            col += 1
        color[q] = col
    ncol = int(color.max()) + 1
    sizes = np.bincount(color)
    out[ds] = dict(c=c, colors=ncol, clusters_per_color_mean=float(sizes.mean()), clusters_per_color_min=int(sizes.min()),
                   clusters_per_color_max=int(sizes.max()))
    print(ds, out[ds], flush=True)
json.dump(out, open(os.path.join(os.environ.get("RES", "."), "coloring.json"), "w"), indent=1)
