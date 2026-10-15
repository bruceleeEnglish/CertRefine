#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单下界（Hamerly 式）证书对照：每点只存一个下界 ℓ_i ≤ min_{k∈N(p)} ‖x_i − μ̄_k‖，
检查时必须减去候选集内的最大位移 max_k δ_k、乘以最小权重 w_min：
    w_min (ℓ_i − max_{k∈N(p)} δ_k)² ≥ key_best  ⇒  跳过全部 K 个候选。
其余（精确起步、顺序 Hartigan、提交校验、随机顺序、种子）与 refine.py 的 cr 完全相同，因此每轮的分配应与 cr 逐位一致，
只有跳过的候选对数不同。用法: refine_single.py <ds> [--seed 42] [--passes 5] [--K 64]
输出: <res>/<ds>/single_s<seed>.refine.json"""
import argparse
import os
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine as R  # noqa: E402
from common import DATASETS, c_of, cpp_train_s, dump_json, labels_digest, read_fvecs, res_dir  # noqa: E402

import faiss  # noqa: E402

TIE = R.TIE
MARGIN = R.MARGIN


@njit(parallel=True, fastmath=True, cache=True)
def decide_single(X, labels, M, cnt, neigh, order, b0, b1, tgt, lb, sb, Lam, pairs_tot, pairs_skip):
    d = X.shape[1]
    K = neigh.shape[1]
    for t in prange(b1 - b0):
        i = order[b0 + t]
        p = labels[i]
        bp = cnt[p]
        best = p
        pt = 0
        ps = 0
        if bp > 1:
            s = 0.0
            for j in range(d):
                u = X[i, j] - M[p, j]
                s += u * u
            bestw = bp / (bp - 1.0) * s
            skipall = False
            if lb[i] > 0.0:
                maxdel = 0.0
                wmin = 1.0
                for ci in range(K):
                    k = neigh[p, ci]
                    if Lam[k] > maxdel:
                        maxdel = Lam[k]
                    wk = cnt[k] / (cnt[k] + 1.0)
                    if wk < wmin:
                        wmin = wk
                lo = sb[i] - maxdel
                if lo > 0.0 and wmin * lo * lo >= bestw:
                    skipall = True
            pt = K
            if skipall:
                ps = K
            else:
                mn = np.inf
                for ci in range(K):
                    k = neigh[p, ci]
                    bk = cnt[k]
                    wk = bk / (bk + 1.0)
                    s = 0.0
                    for j in range(d):
                        u = X[i, j] - M[k, j]
                        s += u * u
                    v = np.sqrt(s) - Lam[k]
                    if v < mn:
                        mn = v
                    wd = wk * s
                    if wd < bestw * (1.0 - TIE):
                        bestw = wd
                        best = k
                sb[i] = mn - MARGIN * abs(mn)
                lb[i] = 1.0
        tgt[t] = best
        pairs_tot[t] = pt
        pairs_skip[t] = ps


@njit(parallel=True, cache=True)
def transfer_single(sb, lb, labels, neigh, shift):
    n = sb.shape[0]
    K = neigh.shape[1]
    for i in prange(n):
        if lb[i] <= 0.0:
            continue
        p = labels[i]
        mx = 0.0
        for ci in range(K):
            v = shift[neigh[p, ci]]
            if v > mx:
                mx = v
        v = sb[i] - mx
        sb[i] = v - MARGIN * abs(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--T", type=int, default=4096)
    a = ap.parse_args()
    ds = a.ds
    c = c_of(ds)
    od = res_dir(ds)
    hp = os.path.join(od, f"hier_s{a.seed}")
    faiss.omp_set_num_threads(20)
    X = read_fvecs(DATASETS[ds][0])
    n, d = X.shape
    M0 = np.ascontiguousarray(read_fvecs(hp + ".centroids.fvecs"), dtype=np.float64)
    neigh = R.build_neigh(M0, a.K)
    index = faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(M0, dtype=np.float32))
    _, I = index.search(X, 1)
    labels = I[:, 0].astype(np.int32)
    A, cnt, M = R.init_state(X, labels, c)
    lb = np.zeros(n)
    sb = np.zeros(n)
    Lam = np.zeros(c)
    moved = np.zeros(n, np.int8)
    rng = np.random.default_rng(a.seed)
    tgt = np.empty(a.T, np.int64)
    pt = np.empty(a.T, np.int64)
    ps = np.empty(a.T, np.int64)
    hist = [dict(epoch=0, labels_md5=labels_digest(labels))]
    for ep in range(1, a.passes + 1):
        t0 = time.perf_counter()
        if ep > 1:
            transfer_single(sb, lb, labels, neigh, Lam)
        Lam[:] = 0.0
        Mcp = M.copy()
        order = rng.permutation(n).astype(np.int64)
        tot = skp = mv = 0
        for b0 in range(0, n, a.T):
            b1 = min(n, b0 + a.T)
            decide_single(X, labels, M, cnt, neigh, order, b0, b1, tgt, lb, sb, Lam, pt, ps)
            m_, _r = R.commit(X, labels, A, M, cnt, order, b0, b1, tgt, lb, Lam, Mcp, False, moved)
            mv += m_
            tot += int(pt[:b1 - b0].sum())
            skp += int(ps[:b1 - b0].sum())
        r = dict(epoch=ep, moves=int(mv), pairs_total=tot, pairs_skipped=skp, skipped_pct=100.0 * skp / max(tot, 1),
                 epoch_s=time.perf_counter() - t0, labels_md5=labels_digest(labels))
        hist.append(r)
        print("ep", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    dump_json(dict(ds=ds, seed=a.seed, K=a.K, cert="single", hist=hist), os.path.join(od, f"single_s{a.seed}.refine.json"))


if __name__ == "__main__":
    main()
