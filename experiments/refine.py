#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CertRefine 及其受控对照。

用法:
  refine.py <ds> --seed 42 --passes 5 [--K 64] [--T 4096] [--rule hartigan|lloyd] [--update seq|batch]
            [--cert 1|0] [--init labels|exact] [--refresh 0] [--tag cr] [--save-every 1]

变体（tag 约定）：
  cr        顺序 Hartigan + 证书（本文方法 CertRefine）
  cr_nocert 顺序 Hartigan、无证书（实测"无证书"时间；并验证与 cr 的分配逐位相同）
  hsl       批量 Lloyd、无证书（Elastic/Lucene 式 shortlist fix-up 的同实现孪生基线）
  hslc      批量 Lloyd + 证书
  sl / hb   顺序 Lloyd + 证书 / 批量 Hartigan + 证书（消融）
  cr_exact  从精确 n×c 分配起步（计时计入），用于量化"起步标签"的影响

计时口径（全部计入 total_s）：分层版训练时间（读 C++ 日志的 train_ms）+ 由标签求簇和/簇大小
+（--init exact 时）精确分配 + 候选图构建 + 每轮的证书搬移、随机置换、并行决策与串行提交校验。
不计入（与所有方法一致）：读数据文件、numba JIT 编译（计时前已预热）。
"""
import argparse
import os
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, c_of, cpp_train_s, dump_json, labels_digest, read_fvecs, res_dir  # noqa: E402

import faiss  # noqa: E402

MARGIN = 1e-6      # 证书下界的相对安全余量：浮点舍入只会让界更松，不会让它失效
TIE = 1e-7         # 目标簇必须在相对意义上严格更优才移动（与原型一致）


# ----------------------------------------------------------------------------- 状态初始化
@njit(cache=True)
def _counting_order(labels, c):
    n = labels.shape[0]
    cnt = np.zeros(c, np.int64)
    for i in range(n):
        cnt[labels[i]] += 1
    starts = np.zeros(c + 1, np.int64)
    for k in range(c):
        starts[k + 1] = starts[k] + cnt[k]
    pos = starts[:-1].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        k = labels[i]
        order[pos[k]] = i
        pos[k] += 1
    return order, starts, cnt


@njit(parallel=True, cache=True)
def _segment_sums(X, order, starts, A):
    c, d = A.shape
    for k in prange(c):
        for t in range(starts[k], starts[k + 1]):
            i = order[t]
            for j in range(d):
                A[k, j] += X[i, j]


def init_state(X, labels, c):
    order, starts, cnt = _counting_order(labels, c)
    A = np.zeros((c, X.shape[1]), np.float64)
    _segment_sums(X, order, starts, A)
    M = A / np.maximum(cnt, 1)[:, None]
    return A, cnt, M


@njit(parallel=True, fastmath=True, cache=True)
def snap(X, labels, M0, neigh0, out):
    """候选内重分配（--init snap）：每点只与其分层簇 p 及 p 的 K 个近邻质心比较，取最近者。
    代价 n(K+1)d，远小于精确分配的 n·c·d；最近质心落在候选集内时与精确分配相同。"""
    n, d = X.shape
    K = neigh0.shape[1]
    for i in prange(n):
        p = labels[i]
        s = 0.0
        for j in range(d):
            u = X[i, j] - M0[p, j]
            s += u * u
        best = p
        bs = s
        for ci in range(K):
            k = neigh0[p, ci]
            s = 0.0
            for j in range(d):
                u = X[i, j] - M0[k, j]
                s += u * u
            if s < bs:
                bs = s
                best = k
        out[i] = best


@njit(parallel=True, fastmath=True, cache=True)
def greedy_assign(X, labels, M0, neigh0, out, hops):
    """质心近邻图上的贪心下山（--init greedy）：从分层标签出发，反复移到 K 个近邻中更近的质心，直到不再改进。
    即在 c 个质心上做一次图搜索；每跳 K 次距离计算，通常 2–3 跳收敛。"""
    n, d = X.shape
    K = neigh0.shape[1]
    for i in prange(n):
        cur = labels[i]
        sc = 0.0
        for j in range(d):
            u = X[i, j] - M0[cur, j]
            sc += u * u
        h = 0
        while h < 32:
            best = cur
            bs = sc
            for ci in range(K):
                k = neigh0[cur, ci]
                s = 0.0
                for j in range(d):
                    u = X[i, j] - M0[k, j]
                    s += u * u
                if s < bs:
                    bs = s
                    best = k
            h += 1
            if best == cur:
                break
            cur = best
            sc = bs
        out[i] = cur
        hops[i] = h


@njit(parallel=True, fastmath=True, cache=True)
def two_level_assign(X, G, gstart, gmem, M0, neigh0, m, out):
    """两级近似最近质心分配（--init ivf）：先在 g≈√c 个组中心里取最近的 m 组，扫描这些组的成员质心；
    再在候选图上补一跳（与该质心的 K 个近邻比较）。每点约 g + m·c/g + K 次距离计算。"""
    n, d = X.shape
    g = G.shape[0]
    K = neigh0.shape[1]
    for i in prange(n):
        bd = np.full(m, np.inf)
        bg = np.full(m, -1, np.int64)
        for q in range(g):
            s = 0.0
            for j in range(d):
                u = X[i, j] - G[q, j]
                s += u * u
            if s < bd[m - 1]:
                t = m - 1
                while t > 0 and bd[t - 1] > s:
                    bd[t] = bd[t - 1]
                    bg[t] = bg[t - 1]
                    t -= 1
                bd[t] = s
                bg[t] = q
        best = -1
        bs = np.inf
        for t in range(m):
            q = bg[t]
            if q < 0:
                continue
            for r in range(gstart[q], gstart[q + 1]):
                k = gmem[r]
                s = 0.0
                for j in range(d):
                    u = X[i, j] - M0[k, j]
                    s += u * u
                if s < bs:
                    bs = s
                    best = k
        cur = best
        for ci in range(K):          # 候选图补一跳
            k = neigh0[cur, ci]
            s = 0.0
            for j in range(d):
                u = X[i, j] - M0[k, j]
                s += u * u
            if s < bs:
                bs = s
                best = k
        out[i] = best


def centroid_groups(M0, seed):
    """把 c 个质心聚成 g=round(√c) 组（FAISS k-means，10 轮，只作用在 c 个质心上，代价可忽略）。"""
    c, d = M0.shape
    g = int(round(np.sqrt(c)))
    km = faiss.Kmeans(d, g, niter=10, seed=seed, verbose=False)
    M0f = np.ascontiguousarray(M0, dtype=np.float32)
    km.train(M0f)
    _, gi = km.index.search(M0f, 1)
    gi = gi[:, 0]
    order = np.argsort(gi, kind="stable").astype(np.int64)
    gstart = np.zeros(g + 1, np.int64)
    np.add.at(gstart, gi + 1, 1)
    gstart = np.cumsum(gstart)
    return km.centroids.astype(np.float64), gstart, order


@njit(cache=True)
def _bucket_by_group(Tg, g):
    """把 (点, 组) 对按组做计数排序：返回每组的点下标（CSR）。Tg: n×m 的组编号。"""
    n, m = Tg.shape
    cnt = np.zeros(g + 1, np.int64)
    for i in range(n):
        for t in range(m):
            cnt[Tg[i, t] + 1] += 1
    for q in range(g):
        cnt[q + 1] += cnt[q]
    pos = cnt[:-1].copy()
    pts = np.empty(n * m, np.int64)
    for i in range(n):
        for t in range(m):
            q = Tg[i, t]
            pts[pos[q]] = i
            pos[q] += 1
    return cnt, pts


def ivf_gemm_assign(X, M0, m, seed):
    """两级近似最近质心分配（--init ivfgemm）：c 个质心先聚成 g≈√c 组；每个点取最近的 m 个组中心；
    然后逐组用一次 BLAS 矩阵乘，计算"选中该组的所有点"到该组成员质心的距离，保留每点的最小者。
    计算量约为精确分配的 m/√c，且全部走 GEMM。"""
    n, d = X.shape
    G, gstart, gmem = centroid_groups(M0, seed)
    g = G.shape[0]
    Gf = np.ascontiguousarray(G, dtype=np.float32)
    gi = faiss.IndexFlatL2(d)
    gi.add(Gf)
    _, Tg = gi.search(X, m)
    starts, pts = _bucket_by_group(Tg.astype(np.int64), g)
    M0f = np.ascontiguousarray(M0, dtype=np.float32)
    cn = (M0f.astype(np.float64) ** 2).sum(1).astype(np.float32)
    best_d = np.full(n, np.inf, np.float32)
    best_k = np.zeros(n, np.int32)
    for q in range(g):
        P = pts[starts[q]:starts[q + 1]]
        mem = gmem[gstart[q]:gstart[q + 1]]
        if P.shape[0] == 0 or mem.shape[0] == 0:
            continue
        D = X[P] @ M0f[mem].T              # |P|×|mem|，BLAS
        D *= -2.0
        D += cn[mem][None, :]              # ‖μ‖² − 2x·μ（‖x‖² 对同一点是常数，比较时可省略）
        j = D.argmin(1)
        dmin = D[np.arange(P.shape[0]), j]
        upd = dmin < best_d[P]
        best_d[P[upd]] = dmin[upd]
        best_k[P[upd]] = mem[j[upd]]
    return best_k


# ----------------------------------------------------------------------------- 证书搬移
@njit(parallel=True, cache=True)
def _transfer(lbk, lb, labels, neigh, shift):
    """轮末把检查点移到当前质心：ℓ' = ℓ − ‖μ̄'_k − μ̄_k‖（= 本轮净位移 δ_k），并留出舍入余量。"""
    n, K = lbk.shape
    for i in prange(n):
        if lb[i] <= 0.0:
            continue
        p = labels[i]
        for ci in range(K):
            v = lbk[i, ci] - shift[neigh[p, ci]]
            lbk[i, ci] = v - MARGIN * abs(v)


@njit(parallel=True, cache=True)
def _remap_slots(lbk, lb, labels, has, pos):
    """候选图刷新：仍在候选集里的候选把界搬到新槽位，新进入的候选界置 0。"""
    n, K = lbk.shape
    for i in prange(n):
        if lb[i] <= 0.0:
            continue
        p = labels[i]
        row = lbk[i].copy()
        for j in range(K):
            lbk[i, j] = row[pos[p, j]] if has[p, j] else 0.0


# ----------------------------------------------------------------------------- 决策与提交
@njit(parallel=True, fastmath=True, cache=True)
def decide(X, labels, M, cnt, neigh, order, b0, b1, tgt, use_cert, lb, lbk, Lam, lloyd,
           pairs_tot, pairs_skip, dims):
    """并行决策（读当前状态，本批内状态不变）。证书：ℓ_{i,j} − δ_k > 0 且 w_k(ℓ_{i,j} − δ_k)² ≥ key_best 时跳过候选 k。"""
    d = X.shape[1]
    K = neigh.shape[1]
    for t in prange(b1 - b0):
        i = order[b0 + t]
        p = labels[i]
        bp = cnt[p]
        best = p
        pt = 0
        ps = 0
        dm = 0.0
        if bp > 1:
            certok = use_cert and lb[i] > 0.0
            s = 0.0
            for j in range(d):
                u = X[i, j] - M[p, j]
                s += u * u
            dm += d
            wp = 1.0 if lloyd else bp / (bp - 1.0)
            bestw = wp * s
            for ci in range(K):
                k = neigh[p, ci]
                bk = cnt[k]
                wk = 1.0 if lloyd else bk / (bk + 1.0)
                pt += 1
                if certok:
                    lo = lbk[i, ci] - Lam[k]
                    if lo > 0.0 and wk * lo * lo >= bestw:
                        ps += 1
                        continue
                s = 0.0
                for j in range(d):
                    u = X[i, j] - M[k, j]
                    s += u * u
                dm += d
                v = np.sqrt(s) - Lam[k]
                lbk[i, ci] = v - MARGIN * abs(v)
                wd = wk * s
                if wd < bestw * (1.0 - TIE):
                    bestw = wd
                    best = k
            lb[i] = 1.0
        tgt[t] = best
        pairs_tot[t] = pt
        pairs_skip[t] = ps
        dims[t] = dm


@njit(fastmath=True, cache=True)
def commit(X, labels, A, M, cnt, order, b0, b1, tgt, lb, Lam, Mcp, lloyd, moved):
    """串行提交：对当前状态重算两个距离，只有目标函数严格下降才提交（单调性定理）。"""
    d = X.shape[1]
    moves = 0
    rejected = 0
    for t in range(b1 - b0):
        i = order[b0 + t]
        q = tgt[t]
        p = labels[i]
        if q == p:
            continue
        if cnt[p] <= 1:
            rejected += 1
            continue
        sp = 0.0
        sq = 0.0
        for j in range(d):
            u = X[i, j] - M[p, j]
            sp += u * u
            v = X[i, j] - M[q, j]
            sq += v * v
        keyp = sp if lloyd else (cnt[p] / (cnt[p] - 1.0)) * sp
        keyq = sq if lloyd else (cnt[q] / (cnt[q] + 1.0)) * sq
        if keyq < keyp * (1.0 - TIE):
            lb[i] = 0.0
            moved[i] = 1
            for j in range(d):
                A[p, j] -= X[i, j]
                A[q, j] += X[i, j]
            cnt[p] -= 1
            cnt[q] += 1
            ip = 1.0 / cnt[p]
            iq = 1.0 / cnt[q]
            s1 = 0.0
            s2 = 0.0
            for j in range(d):
                M[p, j] = A[p, j] * ip
                M[q, j] = A[q, j] * iq
                a1 = M[p, j] - Mcp[p, j]
                s1 += a1 * a1
                a2 = M[q, j] - Mcp[q, j]
                s2 += a2 * a2
            Lam[p] = np.sqrt(s1)
            Lam[q] = np.sqrt(s2)
            labels[i] = q
            moves += 1
        else:
            rejected += 1
    return moves, rejected


@njit(cache=True)
def apply_batch(X, labels, A, cnt, idx, tg, lb, moved, touched):
    """批量语义：整轮用轮初状态决策，轮末统一应用（Lloyd 式）；不会把簇搬空。"""
    d = X.shape[1]
    moves = 0
    for t in range(idx.shape[0]):
        i = idx[t]
        q = tg[t]
        p = labels[i]
        if q == p or cnt[p] <= 1:
            continue
        for j in range(d):
            A[p, j] -= X[i, j]
            A[q, j] += X[i, j]
        cnt[p] -= 1
        cnt[q] += 1
        labels[i] = q
        lb[i] = 0.0
        moved[i] = 1
        touched[p] = 1
        touched[q] = 1
        moves += 1
    return moves


@njit(parallel=True, cache=True)
def recompute_touched(A, cnt, M, Mcp, Lam, touched):
    c, d = A.shape
    for k in prange(c):
        if touched[k] == 0:
            continue
        ik = 1.0 / max(cnt[k], 1)
        s = 0.0
        for j in range(d):
            M[k, j] = A[k, j] * ik
            a = M[k, j] - Mcp[k, j]
            s += a * a
        Lam[k] = np.sqrt(s)


# ----------------------------------------------------------------------------- 候选图
def build_neigh(M, K):
    Mf = np.ascontiguousarray(M, dtype=np.float32)
    index = faiss.IndexFlatL2(Mf.shape[1])
    index.add(Mf)
    _, I = index.search(Mf, K + 1)
    c = Mf.shape[0]
    out = np.empty((c, K), np.int64)
    for p in range(c):
        row = I[p][I[p] != p]
        out[p] = row[:K]
    return out


# ----------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--rule", choices=["hartigan", "lloyd"], default="hartigan")
    ap.add_argument("--update", choices=["seq", "batch"], default="seq")
    ap.add_argument("--cert", type=int, default=1)
    ap.add_argument("--init", choices=["labels", "snap", "greedy", "ivf", "ivfgemm", "ann", "exact"], default="exact",
                    help="labels: 分层版的 meso 内分配；snap: 分层标签 + 候选内重分配（一跳）；"
                         "greedy: 在质心 K 近邻图上从分层标签贪心下山直到收敛；"
                         "ivf: 两级近似最近质心分配（√c 组、探测 m 组，再沿候选图补一跳）；"
                         "ann: FAISS IVF 近似分配；exact: 精确 n×c 分配（默认，一次 GEMM，计时计入）")
    ap.add_argument("--ann-nprobe", type=int, default=8, help="ann/ivf 探测的组数 m")
    ap.add_argument("--diag-exact", action="store_true", help="（不计时）额外算精确分配，报告起步分配与之的一致率")
    ap.add_argument("--refresh", type=int, default=0, help="每 R 轮重建候选图，0 = 只在开始建一次")
    ap.add_argument("--tag", default="cr")
    ap.add_argument("--save-every", type=int, default=1, help="1 = 每轮都存质心；0 = 只存第 1 轮与最后一轮；-1 = 不存（只测时间/验证分配）")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NUMBA_NUM_THREADS", "20")))
    ap.add_argument("--save-labels", action="store_true",
                    help="alongside every saved centroid file also save the refined labels (<tag>_s<seed>_ep<e>.labels.u32): "
                         "the evaluators then build the lists from these labels instead of a final exact assignment")
    ap.add_argument("--final", choices=["none", "snap"], default="none",
                    help="snap: after the last pass, a timed candidate-restricted final assignment (each point to the nearest of "
                         "its list centroid and its K neighbours; n(K+1)d instead of ncd), saved as <tag>_s<seed>_ep<P>snap.*")
    a = ap.parse_args()

    ds = a.ds
    c = c_of(ds)
    out_dir = res_dir(ds)
    hier_prefix = os.path.join(out_dir, f"hier_s{a.seed}")
    faiss.omp_set_num_threads(a.threads)
    lloyd = a.rule == "lloyd"
    batch = a.update == "batch"
    use_cert = bool(a.cert)

    X = read_fvecs(DATASETS[ds][0])
    n, d = X.shape
    hier_s = cpp_train_s(hier_prefix + ".log")
    print(f"ds={ds} n={n} d={d} c={c} K={a.K} T={a.T} rule={a.rule} update={a.update} cert={a.cert} "
          f"init={a.init} refresh={a.refresh} seed={a.seed} tag={a.tag} hier_s={hier_s:.3f}", flush=True)

    # ---- 预热 JIT（不计时）
    _w = np.ascontiguousarray(X[:256])
    _lab = (np.arange(256) % 4).astype(np.int32)
    _A, _cnt, _M = init_state(_w, _lab, 4)
    _nb = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], np.int64)
    _o = np.arange(256, dtype=np.int64)
    _buf = [np.zeros(256, np.int64) for _ in range(3)]
    _lbk = np.zeros((256, 3), np.float32)
    _lb = np.zeros(256)
    _Lam = np.zeros(4)
    decide(_w, _lab, _M, _cnt, _nb, _o, 0, 256, _buf[0], True, _lb, _lbk, _Lam, lloyd, _buf[1], _buf[2], np.zeros(256))
    commit(_w, _lab.copy(), _A.copy(), _M.copy(), _cnt.copy(), _o, 0, 256, _buf[0], _lb.copy(), _Lam.copy(), _M.copy(), lloyd, np.zeros(256, np.int8))
    apply_batch(_w, _lab.copy(), _A.copy(), _cnt.copy(), _o, _buf[0], _lb.copy(), np.zeros(256, np.int8), np.zeros(4, np.int8))
    recompute_touched(_A, _cnt, _M, _M.copy(), _Lam, np.ones(4, np.int8))
    _transfer(_lbk, _lb, _lab, _nb, _Lam)
    _remap_slots(_lbk, _lb, _lab, np.ones((4, 3), np.bool_), np.zeros((4, 3), np.int64))
    snap(_w, _lab, _M, _nb, np.empty(256, np.int32))
    greedy_assign(_w, _lab, _M, _nb, np.empty(256, np.int32), np.empty(256, np.int32))
    two_level_assign(_w, _M[:2].copy(), np.array([0, 2, 4], np.int64), np.arange(4, dtype=np.int64), _M, _nb, 2, np.empty(256, np.int32))
    _bucket_by_group(np.zeros((4, 2), np.int64), 3)

    # ---- 候选图（在分层质心上建一次；起步分配与各轮精修共用，--refresh 时才重建）
    timing = dict(hier_s=hier_s)
    M0 = np.ascontiguousarray(read_fvecs(hier_prefix + ".centroids.fvecs"), dtype=np.float64)
    t0 = time.perf_counter()
    neigh = build_neigh(M0, a.K)
    timing["graph0_s"] = time.perf_counter() - t0

    # ---- 起步分配
    init_moved = 0
    init_hops = None
    hier_labels = np.fromfile(hier_prefix + ".labels.u32", dtype=np.uint32).astype(np.int32)
    assert hier_labels.shape[0] == n and hier_labels.max() < c, (hier_labels.shape, n, c)
    t0 = time.perf_counter()
    if a.init == "labels":
        labels = hier_labels.copy()
    elif a.init in ("snap", "greedy"):
        labels = np.empty(n, np.int32)
        if a.init == "snap":
            snap(X, hier_labels, M0, neigh, labels)
        else:
            hops = np.empty(n, np.int32)
            greedy_assign(X, hier_labels, M0, neigh, labels, hops)
            init_hops = dict(mean=float(hops.mean()), p99=float(np.percentile(hops, 99)), max=int(hops.max()))
    elif a.init == "ivf":
        G, gstart, gmem = centroid_groups(M0, a.seed)
        labels = np.empty(n, np.int32)
        two_level_assign(X, G, gstart, gmem, M0, neigh, a.ann_nprobe, labels)
    elif a.init == "ivfgemm":
        labels = ivf_gemm_assign(X, M0, a.ann_nprobe, a.seed)
    elif a.init == "ann":
        M0f = np.ascontiguousarray(M0, dtype=np.float32)
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, int(round(np.sqrt(c))))
        ivf.cp.seed = a.seed
        ivf.train(M0f)
        ivf.add(M0f)
        ivf.nprobe = a.ann_nprobe
        _, I = ivf.search(X, 1)
        labels = I[:, 0].astype(np.int32)
    else:
        index = faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(M0, dtype=np.float32))
        _, I = index.search(X, 1)
        labels = I[:, 0].astype(np.int32)
    timing["init_assign_s"] = time.perf_counter() - t0
    init_moved = int((labels != hier_labels).sum())
    diag = {}
    if a.diag_exact:   # 诊断（不计时）：起步分配与精确最近质心分配的一致率
        M0f = np.ascontiguousarray(read_fvecs(hier_prefix + ".centroids.fvecs"), dtype=np.float32)
        fi = faiss.IndexFlatL2(d)
        fi.add(M0f)
        _, Ie = fi.search(X, 1)
        diag = dict(agree_exact=float((labels == Ie[:, 0]).mean()), hier_labels_agree_exact=float((hier_labels == Ie[:, 0]).mean()))
        print("diag", diag, flush=True)
    t0 = time.perf_counter()
    A, cnt, M = init_state(X, labels, c)
    lbk = np.zeros((n, a.K), np.float32)     # 全 0：满足不变式，但什么也不证明
    lb = np.zeros(n)                          # lb[i] > 0 表示点 i 的逐候选界有效
    Lam = np.zeros(c)                         # δ_k：相对本轮检查点的净位移
    moved = np.zeros(n, np.int8)
    timing["setup_s"] = time.perf_counter() - t0
    empty0 = int((cnt == 0).sum())

    rng = np.random.default_rng(a.seed)
    total = hier_s + timing["graph0_s"] + timing["init_assign_s"] + timing["setup_s"]
    hist = [dict(epoch=0, total_s=total, empty=empty0, init_moved=init_moved, init_hops=init_hops,
                 labels_md5=labels_digest(labels))]
    print("init", a.init, {k: round(v, 4) for k, v in timing.items()}, "moved", init_moved, "hops", init_hops, flush=True)
    tgt = np.empty(a.T, np.int64)
    pt = np.empty(a.T, np.int64)
    ps = np.empty(a.T, np.int64)
    dm = np.empty(a.T, np.float64)
    for ep in range(1, a.passes + 1):
        t_ep = time.perf_counter()
        # 候选图：开始时已在分层质心上建好；refresh>0 时每 R 轮用当前质心重建（证书按槽位搬移，新候选界置 0）
        t0 = time.perf_counter()
        rebuilt = False
        if a.refresh > 0 and ep > 1 and (ep - 1) % a.refresh == 0:
            new = build_neigh(M, a.K)
            if use_cert:
                eq = new[:, :, None] == neigh[:, None, :]
                has = eq.any(axis=2)
                pos = eq.argmax(axis=2).astype(np.int64)
                _remap_slots(lbk, lb, labels, has, pos)
            neigh = new
            rebuilt = True
        t_graph = time.perf_counter() - t0
        # 证书搬移到新检查点（第 1 轮 δ = 0，无操作）
        t0 = time.perf_counter()
        if use_cert and ep > 1:
            _transfer(lbk, lb, labels, neigh, Lam)
        Lam[:] = 0.0
        Mcp = M.copy()
        order = rng.permutation(n).astype(np.int64)
        moved[:] = 0
        t_prep = time.perf_counter() - t0
        # 决策 + 提交
        t0 = time.perf_counter()
        moves = rejected = 0
        pairs_tot = pairs_skip = 0
        dims = 0.0
        t_dec = t_com = 0.0   # 分阶段计时：并行决策 / 串行提交（批量版为最后的统一应用）
        if not batch:
            for b0 in range(0, n, a.T):
                b1 = min(n, b0 + a.T)
                ta = time.perf_counter()
                decide(X, labels, M, cnt, neigh, order, b0, b1, tgt, use_cert, lb, lbk, Lam, lloyd, pt, ps, dm)
                tb = time.perf_counter()
                m_, r_ = commit(X, labels, A, M, cnt, order, b0, b1, tgt, lb, Lam, Mcp, lloyd, moved)
                t_dec += tb - ta
                t_com += time.perf_counter() - tb
                moves += m_
                rejected += r_
                pairs_tot += int(pt[:b1 - b0].sum())
                pairs_skip += int(ps[:b1 - b0].sum())
                dims += float(dm[:b1 - b0].sum())
        else:
            tg_all = np.empty(n, np.int64)
            for b0 in range(0, n, a.T):
                b1 = min(n, b0 + a.T)
                ta = time.perf_counter()
                decide(X, labels, M, cnt, neigh, order, b0, b1, tgt, use_cert, lb, lbk, Lam, lloyd, pt, ps, dm)
                t_dec += time.perf_counter() - ta
                tg_all[b0:b1] = tgt[:b1 - b0]
                pairs_tot += int(pt[:b1 - b0].sum())
                pairs_skip += int(ps[:b1 - b0].sum())
                dims += float(dm[:b1 - b0].sum())
            tb = time.perf_counter()
            touched = np.zeros(c, np.int8)
            moves = apply_batch(X, labels, A, cnt, order, tg_all, lb, moved, touched)
            recompute_touched(A, cnt, M, Mcp, Lam, touched)
            t_com += time.perf_counter() - tb
        t_pass = time.perf_counter() - t0
        t_ep = time.perf_counter() - t_ep
        total += t_ep
        dig = labels_digest(labels)   # 不计时
        if a.save_every == 1 or (a.save_every == 0 and ep in (1, a.passes)):
            if a.save_labels:   # 先写标签再写质心：评测进程按质心文件的出现时间触发，此时标签已就绪
                labels.astype(np.uint32).tofile(os.path.join(out_dir, f"{a.tag}_s{a.seed}_ep{ep}.labels.u32"))
            np.save(os.path.join(out_dir, f"{a.tag}_s{a.seed}_ep{ep}.centroids.npy"), M.astype(np.float32))
        r = dict(epoch=ep, moves=int(moves), rejected=int(rejected), pairs_total=int(pairs_tot),
                 pairs_skipped=int(pairs_skip), skipped_pct=100.0 * pairs_skip / max(pairs_tot, 1),
                 dims_per_point=dims / n, graph_s=t_graph, graph_rebuilt=rebuilt, prep_s=t_prep,
                 pass_s=t_pass, decide_s=t_dec, commit_s=t_com, epoch_s=t_ep, total_s=total,
                 empty=int((cnt == 0).sum()), labels_md5=dig)
        hist.append(r)
        print("ep", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if moves == 0 and not batch:
            print("converged (no committed move)", flush=True)
    if a.final == "snap":   # 候选受限的最终分配（计时）：替代 n×c 的精确分配；列表不再是 Voronoi 单元，评测按标签建表
        t0 = time.perf_counter()
        snap_labels = np.empty(n, np.int32)
        snap(X, labels, M, neigh, snap_labels)
        timing["final_snap_s"] = time.perf_counter() - t0
        timing["final_snap_moved"] = int((snap_labels != labels).sum())
        snap_labels.astype(np.uint32).tofile(os.path.join(out_dir, f"{a.tag}_s{a.seed}_ep{a.passes}snap.labels.u32"))
        np.save(os.path.join(out_dir, f"{a.tag}_s{a.seed}_ep{a.passes}snap.centroids.npy"), M.astype(np.float32))
        print("final snap", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in timing.items() if k.startswith("final")}, flush=True)
    if a.diag_exact:   # 诊断（不计时）：精修结束时，有多少点落在自己质心的 Voronoi 单元里（IVF 建表用的正是 Voronoi 单元）
        fi = faiss.IndexFlatL2(d)
        fi.add(np.ascontiguousarray(M, dtype=np.float32))
        _, If = fi.search(X, 1)
        diag["final_voronoi_agree"] = float((labels == If[:, 0]).mean())
        print("diag final", {k: round(v, 6) for k, v in diag.items()}, flush=True)
    out = dict(ds=ds, n=n, d=d, c=c, K=a.K, T=a.T, rule=a.rule, update=a.update, cert=use_cert, init=a.init, final=a.final,
               ann_nprobe=a.ann_nprobe, refresh=a.refresh, seed=a.seed, tag=a.tag, threads=a.threads,
               timing=timing, diag=diag, hist=hist)
    dump_json(out, os.path.join(out_dir, f"{a.tag}_s{a.seed}.refine.json"))
    print("saved", a.tag, flush=True)


if __name__ == "__main__":
    main()
