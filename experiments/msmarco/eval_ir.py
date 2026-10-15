#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MS MARCO 端到端检索评测：给定 IVF 质心，在等扫描预算下报告 MRR@10 / nDCG@10 / Recall@100（对 qrels），
以及 ANN recall@10（对精确 top-10）。所有 6,980 条 dev 查询。
用法（$PY，CPU faiss）: eval_ir.py [--names ...] [--force]
输出: <res>/msmarco/<name>.ir.json 与 <name>.ir_pq.npz（逐查询指标，用于配对检验）"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from common import DATA_MSMARCO, DATASETS, dump_json, load_centroids, read_fvecs, read_ivecs, res_dir  # noqa: E402
from eval_scan import BUDGETS, NPROBES, build_ivf, interp  # noqa: E402
from ir_metrics import per_query  # noqa: E402

import faiss  # noqa: E402

METRICS = ["mrr10", "ndcg10", "recall100", "ann_recall10"]


def evaluate(cen, base, Q, gt10, rel):
    n = base.shape[0]
    t0 = time.perf_counter()
    index, quant, sizes, wcss = build_ivf(cen, base)
    c = sizes.shape[0]
    nps = [p for p in NPROBES if p <= c]
    _, L = quant.search(Q, max(nps))
    frac = []
    curves = {m: [] for m in METRICS}
    pq_all = {m: np.zeros((Q.shape[0], len(nps)), np.float16) for m in METRICS}
    for j, npr in enumerate(nps):
        index.nprobe = npr
        _, I = index.search(Q, 100)
        frac.append(float(sizes[L[:, :npr]].sum(1).mean() / n))
        pq = per_query(I, rel)
        pq["ann_recall10"] = np.array([len(np.intersect1d(I[i, :10], gt10[i])) / 10.0 for i in range(Q.shape[0])])
        for m in METRICS:
            curves[m].append(float(pq[m].mean()))
            pq_all[m][:, j] = pq[m]
    out = dict(cv=float(sizes.std() / sizes.mean()), max_list=int(sizes.max()), wcss=wcss, nprobes=nps, scan_frac=frac,
               curves=curves, eval_s=time.perf_counter() - t0)
    for b in BUDGETS:
        for m in METRICS:
            out[f"{m}@scan{b}"] = interp(frac, curves[m], b)
    return out, pq_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=20)
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    od = res_dir("msmarco")
    files = sorted(glob.glob(os.path.join(od, "*.centroids.fvecs")) + glob.glob(os.path.join(od, "*.centroids.npy")))
    todo = []
    for f in files:
        nm = os.path.basename(f).split(".centroids")[0]
        if a.names and nm not in a.names:
            continue
        if a.force or not os.path.exists(os.path.join(od, nm + ".ir.json")):
            todo.append((nm, f))
    if not todo:
        print("msmarco: nothing to evaluate")
        return
    base = read_fvecs(DATASETS["msmarco"][0])
    Q = read_fvecs(DATASETS["msmarco"][1])
    gt10 = read_ivecs(DATASETS["msmarco"][2])[:, :10]
    rel = json.load(open(os.path.join(DATA_MSMARCO, "msmarco_qrels.json")))["rel"]
    for nm, f in todo:
        res, pq = evaluate(load_centroids(f), base, Q, gt10, rel)
        np.savez_compressed(os.path.join(od, nm + ".ir_pq.npz"), **pq)
        dump_json(res, os.path.join(od, nm + ".ir.json"))
        print("msmarco", nm, " ".join("%s=%.4f" % (m, res[f"{m}@scan0.01"]) for m in METRICS), "cv=%.3f (%.0fs)" % (res["cv"], res["eval_s"]), flush=True)


if __name__ == "__main__":
    main()
