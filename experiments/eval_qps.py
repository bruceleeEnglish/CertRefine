#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查询吞吐：对每个方法的质心建 IndexIVFFlat（精确分配），扫 nprobe，测批量查询的 QPS（20 线程、预热后计时，
取 3 次的中位数）与 recall@10。计时实验，须在独立的 NUMA 节点上运行。
用法: eval_qps.py <ds> <name...>   输出: <res>/<ds>/<name>.qps.json"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, dump_json, load_centroids, load_queries, read_fvecs, res_dir  # noqa: E402
from eval_scan import build_ivf  # noqa: E402

import faiss  # noqa: E402

NPROBES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def main():
    ds = sys.argv[1]
    names = sys.argv[2:]
    faiss.omp_set_num_threads(20)
    od = res_dir(ds)
    base = read_fvecs(DATASETS[ds][0])
    q, gt10 = load_queries(ds)
    for nm in names:
        out_p = os.path.join(od, nm + ".qps.json")
        if os.path.exists(out_p):
            continue
        f = [p for p in (os.path.join(od, nm + ".centroids.fvecs"), os.path.join(od, nm + ".centroids.npy")) if os.path.exists(p)]
        if not f:
            print("missing", nm, flush=True)
            continue
        index, quant, sizes, _ = build_ivf(load_centroids(f[0]), base)
        rows = []
        for npr in [p for p in NPROBES if p <= sizes.shape[0]]:
            index.nprobe = npr
            index.search(q[:200], 10)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                _, I = index.search(q, 10)
                ts.append(time.perf_counter() - t0)
            rec = float(np.mean([len(np.intersect1d(I[i], gt10[i])) for i in range(q.shape[0])]) / 10.0)
            rows.append(dict(nprobe=npr, qps=q.shape[0] / float(np.median(ts)), recall=rec))
        dump_json(dict(curve=rows, threads=20, nq=int(q.shape[0])), out_p)
        print(ds, nm, [(r["nprobe"], round(r["recall"], 4), int(r["qps"])) for r in rows[3:10:3]], flush=True)


if __name__ == "__main__":
    main()
