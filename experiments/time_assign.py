#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最终列表分配的耗时：把全部 n 个向量精确分配到 c 个质心（FAISS IndexFlatL2，k=1，20 线程）。
这一步对所有方法相同（与质心取值无关），论文的端到端构建时间 = 训练时间 + 本步耗时。
计时实验：须在 node0 上与其他计时实验串行运行。取 3 次的中位数。
用法: time_assign.py <ds...>     输出: $RES/assign_time.json（按数据集合并写入）"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, dump_json, load_centroids, read_fvecs, res_dir  # noqa: E402

import faiss  # noqa: E402

faiss.omp_set_num_threads(20)
out_p = os.path.join(os.environ.get("RES", os.path.join(os.path.dirname(__file__), "..", "results")), "assign_time.json")
res = json.load(open(out_p)) if os.path.exists(out_p) else {}
for ds in sys.argv[1:]:
    od = res_dir(ds)
    cand = [os.path.join(od, f) for f in ("cr_s42_ep5.centroids.npy", "hier_s42.centroids.fvecs") if os.path.exists(os.path.join(od, f))]
    if not cand:
        print("no centroids for", ds)
        continue
    C = np.ascontiguousarray(load_centroids(cand[0]), dtype=np.float32)
    X = read_fvecs(DATASETS[ds][0])
    ts = []
    for _ in range(3):
        idx = faiss.IndexFlatL2(C.shape[1])
        t0 = time.perf_counter()
        idx.add(C)
        idx.search(X, 1)
        ts.append(time.perf_counter() - t0)
    res[ds] = dict(assign_s=float(np.median(ts)), runs=ts, n=int(X.shape[0]), c=int(C.shape[0]))
    print(ds, "final assignment %.2fs (median of 3)" % np.median(ts), flush=True)
    dump_json(res, out_p)
