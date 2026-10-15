#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MS MARCO 端到端评测（GPU，稠密 nprobe 网格，精确等价于 IVFFlat 检索）。

IVFFlat 精确扫描被探测的表：查询在 nprobe=p 时返回的 top-k，正是精确近邻序列中"所在表的名次 < p"的前 k 个。
因此先算一次每条查询的精确 top-N（N=2000，内积 = 余弦，两边都已 L2 归一化），之后对任意质心、任意 p，
IVF 的 top-100 = 精确 top-N 里满足条件的前 100 个——前提是满足条件的至少有 100 个（@10 指标只需 10 个）；
不满足的查询数逐预算记录在输出里（"incomplete"），需要时可用 eval_ir.py（FAISS 真实检索）核对。

输出: <res>/msmarco/<name>.ir.json（稠密曲线与各预算处的指标）、<name>.ir_pqb.npz（逐查询、逐预算的指标，供配对检验）
用法（$PY_ENC，torch）: eval_ir_gpu.py [--names ...] [--force] [--device cuda:0] [--verify name]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import DATA_MSMARCO, DATASETS, dump_json, load_centroids, read_fvecs, res_dir  # noqa: E402

BUDGETS = [0.0025, 0.005, 0.01, 0.02, 0.04]
NTOP = 2000
PMAX = 1024
EVALUATOR = "gpu-dense-ir-v1"
torch.backends.cuda.matmul.allow_tf32 = False


def exact_topn(base_g, Qg, ntop, step=100_000):
    best_s = torch.full((Qg.shape[0], ntop), -1e9, device=Qg.device)
    best_i = torch.zeros((Qg.shape[0], ntop), dtype=torch.int64, device=Qg.device)
    for s in range(0, base_g.shape[0], step):
        sc = Qg @ base_g[s:s + step].T
        vs, ix = torch.topk(sc, min(ntop, sc.shape[1]), dim=1)
        cat_s = torch.cat([best_s, vs], 1)
        cat_i = torch.cat([best_i, ix + s], 1)
        top = torch.topk(cat_s, ntop, dim=1)
        best_s, best_i = top.values, torch.gather(cat_i, 1, top.indices)
    return best_i


def assign(base_g, C, chunk=None):
    chunk = chunk or max(4096, int(1.5e9 // (4 * C.shape[0])))
    cn = (C * C).sum(1)
    labels = torch.empty(base_g.shape[0], dtype=torch.int64, device=base_g.device)
    for s in range(0, base_g.shape[0], chunk):
        D = torch.addmm(cn[None, :], base_g[s:s + chunk], C.T, beta=1.0, alpha=-2.0)
        labels[s:s + chunk] = D.argmin(1)
    return labels


def bracket(frac, b):
    if not (frac[0] <= b <= frac[-1]):
        return None
    j = int(np.searchsorted(frac, b)) - 1
    j = min(max(j, 0), len(frac) - 2)
    return j, ((b - frac[j]) / (frac[j + 1] - frac[j]) if frac[j + 1] > frac[j] else 0.0)


def metrics_at(gr, relm, nrel, p, idcg):
    """gr: nq×N 近邻所在表的名次；relm: nq×N 是否相关；返回逐查询 (mrr10, ndcg10, r100, ann10, 满足条件的个数)。"""
    ok = gr < p
    pos = torch.cumsum(ok.to(torch.int32), dim=1)                     # 在 IVF 结果中的位置（1 起），仅 ok 处有意义
    rel_ok = ok & relm
    in10 = rel_ok & (pos <= 10)
    first = torch.where(in10, pos, torch.full_like(pos, 10 ** 6)).min(1).values
    mrr = torch.where(first <= 10, 1.0 / first.double(), torch.zeros_like(first, dtype=torch.float64))
    disc = torch.where(in10, 1.0 / torch.log2(pos.double() + 1.0), torch.zeros_like(pos, dtype=torch.float64)).sum(1)
    ndcg = disc / idcg
    r100 = (rel_ok & (pos <= 100)).sum(1).double() / nrel.clamp_min(1)
    ann10 = ok[:, :10].sum(1).double() / 10.0
    return mrr, ndcg, r100, ann10, ok.sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-age", type=float, default=0.0)
    a = ap.parse_args()
    dev = torch.device(a.device)
    od = res_dir("msmarco")
    files = sorted(glob.glob(os.path.join(od, "*.centroids.fvecs")) + glob.glob(os.path.join(od, "*.centroids.npy")))
    todo = []
    for f in files:
        nm = os.path.basename(f).split(".centroids")[0]
        if a.names and nm not in a.names:
            continue
        if a.min_age and time.time() - os.path.getmtime(f) < a.min_age:
            continue
        jp = os.path.join(od, nm + ".ir.json")
        done = os.path.exists(jp) and json.load(open(jp)).get("evaluator") == EVALUATOR
        if a.force or not done:
            todo.append((nm, f))
    if not todo:
        print("msmarco: nothing to evaluate", flush=True)
        return
    base = read_fvecs(DATASETS["msmarco"][0])
    n = base.shape[0]
    base_g = torch.from_numpy(base).to(dev)
    del base
    Q = read_fvecs(DATASETS["msmarco"][1])
    Qg = torch.from_numpy(Q).to(dev)
    nq = Q.shape[0]
    top_p = os.path.join(DATA_MSMARCO, f"msmarco_top{NTOP}.npy")
    if os.path.exists(top_p):
        topn = torch.from_numpy(np.load(top_p).astype(np.int64)).to(dev)
    else:
        t0 = time.perf_counter()
        topn = exact_topn(base_g, Qg, NTOP)
        np.save(top_p, topn.cpu().numpy().astype(np.int32))
        print(f"exact top-{NTOP} computed in {time.perf_counter() - t0:.0f}s", flush=True)
    rel = json.load(open(os.path.join(DATA_MSMARCO, "msmarco_qrels.json")))["rel"]
    relm = torch.zeros((nq, NTOP), dtype=torch.bool, device=dev)
    tn = topn.cpu().numpy()
    for qi, r in enumerate(rel):
        if r:
            relm[qi] = torch.from_numpy(np.isin(tn[qi], np.asarray(r))).to(dev)
    nrel = torch.tensor([len(r) for r in rel], dtype=torch.float64, device=dev)
    disc = 1.0 / np.log2(np.arange(2, 12))
    idcg = torch.tensor([disc[:min(len(r), 10)].sum() if r else 1.0 for r in rel], dtype=torch.float64, device=dev)
    for nm, f in todo:
        t0 = time.perf_counter()
        C = torch.from_numpy(np.ascontiguousarray(load_centroids(f), dtype=np.float32)).to(dev)
        c = C.shape[0]
        lf = os.path.join(od, nm + ".labels.u32")
        if os.path.exists(lf):   # 按文件里的标签建表（不做精确分配）
            lab = np.fromfile(lf, dtype=np.uint32).astype(np.int64)
            assert lab.shape[0] == n and lab.max() < c, (lab.shape, n, c)
            labels = torch.from_numpy(lab).to(dev)
        else:
            labels = assign(base_g, C)
        sizes = torch.bincount(labels, minlength=c).double()
        pmax = min(c, PMAX)
        dq = torch.addmm((C * C).sum(1)[None, :], Qg, C.T, beta=1.0, alpha=-2.0)
        L = torch.topk(dq, pmax, dim=1, largest=False, sorted=True).indices
        rank = torch.full((nq, c), pmax, dtype=torch.int64, device=dev)
        rank.scatter_(1, L, torch.arange(pmax, device=dev).expand(nq, pmax))
        gr = torch.gather(rank, 1, labels[topn])                            # nq×N
        csum = torch.cumsum(sizes[L], dim=1)
        frac = (csum.mean(0) / n).cpu().numpy()
        curves = {m: np.zeros(pmax) for m in ("mrr10", "ndcg10", "recall100", "ann_recall10")}
        # 只在各预算两侧的整数 nprobe 上保留逐查询值；曲线在所有 p 上计算
        need = {}
        for b in BUDGETS:
            br = bracket(frac, b)
            if br:
                need[br[0] + 1] = None
                need[br[0] + 2] = None
        per_p = {}
        incomplete = {}
        for p in range(1, pmax + 1):
            mrr, ndcg, r100, ann, cnt = metrics_at(gr, relm, nrel, p, idcg)
            curves["mrr10"][p - 1] = float(mrr.mean())
            curves["ndcg10"][p - 1] = float(ndcg.mean())
            curves["recall100"][p - 1] = float(r100.mean())
            curves["ann_recall10"][p - 1] = float(ann.mean())
            if p in need:
                per_p[p] = dict(mrr10=mrr.cpu().numpy(), ndcg10=ndcg.cpu().numpy(), recall100=r100.cpu().numpy(),
                                ann_recall10=ann.cpu().numpy())
                incomplete[p] = dict(lt100=int((cnt < 100).sum()), lt10=int((cnt < 10).sum()))
        out = dict(cv=float(sizes.std() / sizes.mean()), max_list=int(sizes.max()), nprobes=list(range(1, pmax + 1)),
                   scan_frac=frac.tolist(), curves={k: v.tolist() for k, v in curves.items()}, evaluator=EVALUATOR, ntop=NTOP,
                   assignment="labels" if os.path.exists(lf) else "exact")
        pqb = {m: np.full((nq, len(BUDGETS)), np.nan, np.float32) for m in curves}
        for bi, b in enumerate(BUDGETS):
            br = bracket(frac, b)
            if br is None:
                for m in curves:
                    out[f"{m}@scan{b}"] = float("nan")
                continue
            j, w = br
            for m in curves:
                out[f"{m}@scan{b}"] = float((1 - w) * curves[m][j] + w * curves[m][j + 1])
                pqb[m][:, bi] = (1 - w) * per_p[j + 1][m] + w * per_p[j + 2][m]
            out[f"incomplete@scan{b}"] = max(incomplete[j + 1]["lt100"], incomplete[j + 2]["lt100"])
            out[f"incomplete10@scan{b}"] = max(incomplete[j + 1]["lt10"], incomplete[j + 2]["lt10"])
        out["eval_s"] = time.perf_counter() - t0
        np.savez_compressed(os.path.join(od, nm + ".ir_pqb.npz"), **pqb)
        dump_json(out, os.path.join(od, nm + ".ir.json"))
        print("msmarco", nm, " ".join("%s=%.4f" % (m, out[f"{m}@scan0.01"]) for m in curves),
              "incomplete@1%%=%d cv=%.3f (%.0fs)" % (out["incomplete@scan0.01"], out["cv"], out["eval_s"]), flush=True)


if __name__ == "__main__":
    main()
