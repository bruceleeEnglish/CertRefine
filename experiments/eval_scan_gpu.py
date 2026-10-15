#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""等扫描预算评测（GPU 版，torch float32，关闭 TF32），在**每一个整数 nprobe** 上计算召回（稠密网格）。

原理与 eval_scan.py 的 curve_fast 相同：IVFFlat 精确扫描表时，真实近邻被返回当且仅当它所在的表被探测到，
所以 hits(q, nprobe) = #{g ∈ gt10(q) : rank_q(list(g)) < nprobe}。稠密网格消除了粗网格线性插值带来的偏差
（召回对扫描比例是凹的，粗网格上的弦会低估召回，且各方法低估的程度不同）。

输出（格式向后兼容）：
  <name>.scan.json   nprobes = 1..pmax 的稠密曲线（scan_frac、recall、scan_p99），recall@scan{b}、p99@scan{b}
  <name>.hits.npy    旧粗网格 NPROBES 上的逐查询命中数（uint8），供旧代码使用
  <name>.pqb.npy     逐查询在每个预算 BUDGETS 处的召回（float32，nq×len(BUDGETS)），与均值曲线同一插值权重
用法: eval_scan_gpu.py <ds> [--names ...] [--force] [--min-age 30] [--device cuda:0] [--pmax 1024]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, dump_json, load_centroids, load_queries, read_fvecs, res_dir  # noqa: E402

# 与 eval_scan.py 相同的粗网格（只用于写兼容的 hits.npy）与预算
NPROBES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
BUDGETS = [0.0025, 0.005, 0.01, 0.02, 0.04]
EVALUATOR = "gpu-dense-v2"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def assign(Xg, xn, C, chunk=None):
    """精确最近质心分配：argmin_k ‖c_k‖² − 2 x·c_k；返回标签与 WCSS。分块使每块距离矩阵约 1.5 GB。"""
    chunk = chunk or max(4096, int(1.5e9 // (4 * C.shape[0])))
    cn = (C * C).sum(1)
    labels = torch.empty(Xg.shape[0], dtype=torch.int64, device=Xg.device)
    wcss = 0.0
    for s in range(0, Xg.shape[0], chunk):
        D = torch.addmm(cn[None, :], Xg[s:s + chunk], C.T, beta=1.0, alpha=-2.0)
        v, i = D.min(1)
        labels[s:s + chunk] = i
        wcss += float((v + xn[s:s + chunk]).double().clamp_min(0).sum())
    return labels, wcss


def bracket(frac, b):
    """预算 b 在单调递增的 frac 上的插值位置：返回 (j, w)，使 b = (1-w)·frac[j] + w·frac[j+1]；越界返回 None。"""
    if not (frac[0] <= b <= frac[-1]):
        return None
    j = int(np.searchsorted(frac, b)) - 1
    j = min(max(j, 0), len(frac) - 2)
    w = (b - frac[j]) / (frac[j + 1] - frac[j]) if frac[j + 1] > frac[j] else 0.0
    return j, w


def evaluate(Xg, xn, cen, Q, gt10, dev, pmax_cap=1024, labels_file=None):
    """labels_file: 给出时不做精确分配，直接用文件里的标签建表（"lists as refined" / 候选受限最终分配的协议变体）。"""
    t0 = time.perf_counter()
    n = Xg.shape[0]
    C = torch.from_numpy(np.ascontiguousarray(cen, dtype=np.float32)).to(dev)
    c = C.shape[0]
    if labels_file:
        lab = np.fromfile(labels_file, dtype=np.uint32).astype(np.int64)
        assert lab.shape[0] == n and lab.max() < c, (lab.shape, n, c)
        labels = torch.from_numpy(lab).to(dev)
        wcss = float("nan")
    else:
        labels, wcss = assign(Xg, xn, C)
    sizes = torch.bincount(labels, minlength=c).double()
    pmax = min(c, pmax_cap)
    Qg = torch.from_numpy(Q).to(dev)
    nq = Q.shape[0]
    dq = torch.addmm((C * C).sum(1)[None, :], Qg, C.T, beta=1.0, alpha=-2.0)
    L = torch.topk(dq, pmax, dim=1, largest=False, sorted=True).indices                   # 每条查询的探测顺序
    rank = torch.full((nq, c), pmax, dtype=torch.int64, device=dev)
    rank.scatter_(1, L, torch.arange(pmax, device=dev).expand(nq, pmax))
    g = torch.from_numpy(gt10.astype(np.int64)).to(dev)
    gr = torch.gather(rank, 1, labels[g])                                                 # nq×10：真实近邻所在表的名次
    # hits[q, p-1] = 名次 < p 的真实近邻个数，p = 1..pmax（名次直方图的前缀和）
    hist = torch.zeros((nq, pmax + 1), dtype=torch.int64, device=dev)
    hist.scatter_add_(1, gr.clamp_max(pmax), torch.ones_like(gr))
    hits = torch.cumsum(hist[:, :pmax], dim=1)                                            # nq×pmax
    csum = torch.cumsum(sizes[L], dim=1)                                                  # nq×pmax：探测前 p 个表的扫描量
    frac = (csum.mean(0) / n).cpu().numpy()
    rec = (hits.double().mean(0) / 10.0).cpu().numpy()
    p99 = torch.quantile(csum, 0.99, dim=0).cpu().numpy()
    hits_np = hits.cpu().numpy()
    out = dict(cv=float(sizes.std().item() / sizes.mean().item()), empty=int((sizes == 0).sum().item()),
               max_list=int(sizes.max().item()), p99_list=float(np.percentile(sizes.cpu().numpy(), 99)), wcss=wcss,
               nprobes=list(range(1, pmax + 1)), scan_frac=frac.tolist(), recall=rec.tolist(), scan_p99=p99.tolist(),
               eval_s=0.0, evaluator=EVALUATOR, assignment="labels" if labels_file else "exact")
    pqb = np.full((nq, len(BUDGETS)), np.nan, np.float32)
    for bi, b in enumerate(BUDGETS):
        br = bracket(frac, b)
        if br is None:
            out[f"recall@scan{b}"] = float("nan")
            out[f"p99@scan{b}"] = float("nan")
            continue
        j, w = br
        out[f"recall@scan{b}"] = float((1 - w) * rec[j] + w * rec[j + 1])
        out[f"p99@scan{b}"] = float((1 - w) * p99[j] + w * p99[j + 1]) / n
        pqb[:, bi] = ((1 - w) * hits_np[:, j] + w * hits_np[:, j + 1]) / 10.0
    coarse = [p for p in NPROBES if p <= pmax]
    hits_coarse = hits_np[:, [p - 1 for p in coarse]].astype(np.uint8)
    out["eval_s"] = time.perf_counter() - t0
    return out, hits_coarse, pqb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-age", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pmax", type=int, default=1024)
    a = ap.parse_args()
    dev = torch.device(a.device)
    od = res_dir(a.ds)
    files = sorted(glob.glob(os.path.join(od, "*.centroids.fvecs")) + glob.glob(os.path.join(od, "*.centroids.npy")))
    todo = []
    for f in files:
        name = os.path.basename(f).split(".centroids")[0]
        if a.names and name not in a.names:
            continue
        if a.min_age and time.time() - os.path.getmtime(f) < a.min_age:
            continue
        sp = os.path.join(od, name + ".scan.json")
        done = False
        if os.path.exists(sp) and os.path.exists(os.path.join(od, name + ".pqb.npy")):
            try:
                done = json.load(open(sp)).get("evaluator") == EVALUATOR
            except Exception:
                done = False
        if a.force or not done:
            todo.append((name, f))
    if not todo:
        print(a.ds, "nothing to evaluate", flush=True)
        return
    X = read_fvecs(DATASETS[a.ds][0])
    Xg = torch.from_numpy(X).to(dev)
    xn = (Xg * Xg).sum(1)
    del X
    Q, gt10 = load_queries(a.ds)
    for name, f in todo:
        lf = os.path.join(od, name + ".labels.u32")
        res, hits, pqb = evaluate(Xg, xn, load_centroids(f), Q, gt10, dev, a.pmax, lf if os.path.exists(lf) else None)
        np.save(os.path.join(od, name + ".hits.npy"), hits)
        np.save(os.path.join(od, name + ".pqb.npy"), pqb)
        dump_json(res, os.path.join(od, name + ".scan.json"))
        print(a.ds, name, "r@1%%=%.4f cv=%.3f (%.1fs gpu, dense, %s)" % (res["recall@scan0.01"], res["cv"], res["eval_s"], res["assignment"]), flush=True)


if __name__ == "__main__":
    main()
