#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""等扫描预算评测（IVF 训练方法比较的主协议）。

对一个质心文件：FAISS 精确分配全部 n 个向量建 IndexIVFFlat（所有方法同一做法，差异只来自质心），
扫 nprobe 序列，记录 (平均扫描向量比例, recall@10)，并在扫描预算 0.25%–4% 处线性插值。
同时保存逐查询命中数（用于配对随机化检验）、簇大小统计与精确分配下的 k-means 目标值（WCSS）。

用法: eval_scan.py <ds> [--names n1 n2 ...] [--force] [--threads 20]
  不给 --names 时评测结果目录下所有尚未评测的 *.centroids.fvecs / *.centroids.npy。
输出: <res>/<ds>/<name>.scan.json 与 <name>.hits.npy（uint8, nq × len(NPROBES)）
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, dump_json, load_centroids, load_queries, read_fvecs, res_dir  # noqa: E402

import faiss  # noqa: E402
from faiss.contrib.ivf_tools import add_preassigned  # noqa: E402

NPROBES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
BUDGETS = [0.0025, 0.005, 0.01, 0.02, 0.04]


def build_ivf(cen, base, labels=None):
    """labels=None：精确最近质心分配（静态协议）；否则按给定分配建表（动态维护协议）。"""
    c, d = cen.shape
    quant = faiss.IndexFlatL2(d)
    quant.add(np.ascontiguousarray(cen, dtype=np.float32))
    index = faiss.IndexIVFFlat(quant, d, c)
    index.is_trained = True
    wcss = float("nan")
    if labels is None:
        D, I = quant.search(base, 1)
        labels = I[:, 0]
        wcss = float(D.astype(np.float64).sum())
    add_preassigned(index, base, np.asarray(labels, dtype=np.int64))
    sizes = np.array([index.invlists.list_size(i) for i in range(c)], np.float64)
    return index, quant, sizes, wcss


def curve(index, quant, sizes, q, gt10, n):
    c = sizes.shape[0]
    nps = [p for p in NPROBES if p <= c]
    _, L = quant.search(q, max(nps))
    hits = np.zeros((q.shape[0], len(nps)), np.uint8)
    frac, rec, scan_p99 = [], [], []
    for j, npr in enumerate(nps):
        index.nprobe = npr
        _, I = index.search(q, 10)
        for qi in range(q.shape[0]):
            hits[qi, j] = len(np.intersect1d(I[qi], gt10[qi], assume_unique=False))
        scanned = sizes[L[:, :npr]].sum(1)
        frac.append(float(scanned.mean() / n))
        scan_p99.append(float(np.percentile(scanned, 99)))
        rec.append(float(hits[:, j].mean() / 10.0))
    return nps, hits, frac, rec, scan_p99


def curve_fast(cen, base, q, gt10, labels=None):
    """与 curve() 等价的快速评测。IndexIVFFlat 计算精确距离，因此真实 top-10 近邻 g 被返回，当且仅当 g 所在的列表被探测：
    被扫描且比 g 更近的向量只可能是比 g 更近的真实近邻（至多 9 个），所以 g 在扫描集合中的名次 ≤ 10。
    于是 hits(q, nprobe) = #{g ∈ top10(q) : rank_q(list(g)) < nprobe}，无需执行 IVF 检索；只需一次全量分配求列表与列表大小。"""
    c, d = cen.shape
    n = base.shape[0]
    quant = faiss.IndexFlatL2(d)
    quant.add(np.ascontiguousarray(cen, dtype=np.float32))
    wcss = float("nan")
    if labels is None:
        D, I = quant.search(base, 1)
        labels = I[:, 0]
        wcss = float(D.astype(np.float64).sum())
    labels = np.asarray(labels, dtype=np.int64)
    sizes = np.bincount(labels, minlength=c).astype(np.float64)
    nps = [p for p in NPROBES if p <= c]
    pmax = max(nps)
    _, L = quant.search(q, pmax)                       # 每条查询的质心探测顺序
    rank = np.full((q.shape[0], c), pmax, np.int32)    # 列表在该查询探测顺序中的名次（未进入前 pmax 记为 pmax）
    rows = np.repeat(np.arange(q.shape[0]), pmax)
    rank[rows, L.ravel()] = np.tile(np.arange(pmax, dtype=np.int32), q.shape[0])
    gr = rank[np.arange(q.shape[0])[:, None], labels[gt10]]   # nq×10：真实近邻所在列表的名次
    hits = np.zeros((q.shape[0], len(nps)), np.uint8)
    frac, rec, p99 = [], [], []
    csum = np.cumsum(sizes[L], axis=1)                 # 探测前 t 个列表时扫描的向量数
    for j, npr in enumerate(nps):
        hits[:, j] = (gr < npr).sum(1)
        scanned = csum[:, npr - 1]
        frac.append(float(scanned.mean() / n))
        p99.append(float(np.percentile(scanned, 99)))
        rec.append(float(hits[:, j].mean() / 10.0))
    return sizes, wcss, nps, hits, frac, rec, p99


def interp(frac, rec, b):
    return float(np.interp(b, frac, rec)) if frac[0] <= b <= frac[-1] else float("nan")


def evaluate(cen, base, q, gt10, labels=None, fast=True):
    n = base.shape[0]
    t0 = time.perf_counter()
    if fast:
        sizes, wcss, nps, hits, frac, rec, p99 = curve_fast(cen, base, q, gt10, labels)
    else:
        index, quant, sizes, wcss = build_ivf(cen, base, labels)
        nps, hits, frac, rec, p99 = curve(index, quant, sizes, q, gt10, n)
    out = dict(cv=float(sizes.std() / sizes.mean()), empty=int((sizes == 0).sum()), max_list=int(sizes.max()),
               p99_list=float(np.percentile(sizes, 99)), wcss=wcss, nprobes=nps, scan_frac=frac, recall=rec,
               scan_p99=p99, eval_s=time.perf_counter() - t0)
    for b in BUDGETS:
        out[f"recall@scan{b}"] = interp(frac, rec, b)
    return out, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--min-age", type=float, default=0.0, help="只评测修改时间早于这么多秒的文件（避免读到正在写的文件）")
    ap.add_argument("--slow", action="store_true", help="用真实的 FAISS IVF 检索评测（与默认的快速等价方法对照用）")
    ap.add_argument("--verify", action="store_true", help="对已有 scan.json 用另一种方法重算并报告最大差异，不写文件")
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    od = res_dir(a.ds)
    if a.names:
        files = []
        for nm in a.names:
            f = [p for p in (os.path.join(od, nm + ".centroids.fvecs"), os.path.join(od, nm + ".centroids.npy")) if os.path.exists(p)]
            if not f:
                print("missing", nm, flush=True)
                continue
            files.append(f[0])
    else:
        files = sorted(glob.glob(os.path.join(od, "*.centroids.fvecs")) + glob.glob(os.path.join(od, "*.centroids.npy")))
    todo = []
    for f in files:
        name = os.path.basename(f).split(".centroids")[0]
        if a.min_age and time.time() - os.path.getmtime(f) < a.min_age:
            continue
        done = os.path.exists(os.path.join(od, name + ".scan.json"))
        if a.verify and done:
            todo.append((name, f))
        elif not a.verify and (a.force or not done):
            todo.append((name, f))
    if not todo:
        print(a.ds, "nothing to evaluate", flush=True)
        return
    base = read_fvecs(DATASETS[a.ds][0])
    q, gt10 = load_queries(a.ds)
    if a.verify:
        import json
        for name, f in todo:
            old = json.load(open(os.path.join(od, name + ".scan.json")))
            new, h = evaluate(load_centroids(f), base, q, gt10, fast=not a.slow)
            dr = max(abs(x - y) for x, y in zip(old["recall"], new["recall"]))
            df = max(abs(x - y) for x, y in zip(old["scan_frac"], new["scan_frac"]))
            oh = np.load(os.path.join(od, name + ".hits.npy"))
            print(a.ds, name, "max|d recall|=%.2e max|d frac|=%.2e hits differ in %d cells; %.1fs vs %.1fs" %
                  (dr, df, int((oh != h).sum()), new["eval_s"], old["eval_s"]), flush=True)
        return
    for name, f in todo:
        res, hits = evaluate(load_centroids(f), base, q, gt10, fast=not a.slow)
        np.save(os.path.join(od, name + ".hits.npy"), hits)
        dump_json(res, os.path.join(od, name + ".scan.json"))
        print(a.ds, name, "r@1%%=%.4f cv=%.3f wcss=%.4g (%.1fs)" % (res["recall@scan0.01"], res["cv"], res["wcss"], res["eval_s"]), flush=True)


if __name__ == "__main__":
    main()
