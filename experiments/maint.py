#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""增量插入下的 IVF 维护。

建索引：在初始子集上跑分层 SuperKMeans（C++）→ 精确最近质心分配 → CertRefine 5 轮（与主实验同一流程）。
之后分批插入，比较四种维护策略（各自维护独立状态，建索引结果相同）：
  frozen       质心冻结，新点进最近质心的列表（只追加）
  local_lloyd  新点进最近列表并更新该簇质心；被插入触碰的簇里的点按候选内最近质心重分配一次（无证书；
               近似 SPFresh/LIRE 的"局部重分配"，不含拆分/合并）
  cert_lloyd   新点进最近列表 + 对全体点跑 1 轮带证书的批量 Lloyd（与本文相同的证书机制）
  cert_hart    新点按 Hartigan 判据入簇 + 对全体点跑 1 轮带证书的顺序 Hartigan（本文）
评测（每批后）：用各策略维护出来的列表本身建 IVF（不重新分配），在等扫描预算下测 recall@10，
并记录最大列表、cv、预算 1% 时单查询扫描量的 p99、维护的距离计算量与墙钟时间。

用法: maint.py <ds> --workload uniform|clustered|drift [--seed 42]
工作负载：uniform = 随机 10% 分 10 批插入；clustered = 离 20 个随机种子点最近的 10% 分 10 批插入；
          drift = 按第一主成分排序，后 30% 按序分 6 批插入（渐进漂移）。
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, cpp_train_s, dump_json, read_fvecs, res_dir, write_fvecs  # noqa: E402
from eval_scan import BUDGETS, NPROBES, build_ivf, interp  # noqa: E402
import refine as R  # noqa: E402
from numba import njit, prange  # noqa: E402

import faiss  # noqa: E402

K, T = 64, 4096


@njit(parallel=True, fastmath=True, cache=True)
def cand_nearest(X, pts, labels, M, neigh, out):
    """local_lloyd：pts 中每个点在 {p}∪N(p) 里取最近质心。"""
    d = X.shape[1]
    Kk = neigh.shape[1]
    for t in prange(pts.shape[0]):
        i = pts[t]
        p = labels[i]
        bs = 0.0
        for j in range(d):
            u = X[i, j] - M[p, j]
            bs += u * u
        best = p
        for ci in range(Kk):
            k = neigh[p, ci]
            s = 0.0
            for j in range(d):
                u = X[i, j] - M[k, j]
                s += u * u
            if s < bs:
                bs = s
                best = k
        out[t] = best


@njit(parallel=True, fastmath=True, cache=True)
def hartigan_pick(Xn, p0, M, cnt, neigh, out):
    """新点入簇（cert_hart）：最近质心 p0 与其 K 近邻中取 w_k‖x−μ_k‖² 最小者（w_k = n_k/(n_k+1)）。"""
    d = Xn.shape[1]
    Kk = neigh.shape[1]
    for t in prange(Xn.shape[0]):
        p = p0[t]
        s = 0.0
        for j in range(d):
            u = Xn[t, j] - M[p, j]
            s += u * u
        bw = (cnt[p] / (cnt[p] + 1.0)) * s
        best = p
        for ci in range(Kk):
            k = neigh[p, ci]
            s = 0.0
            for j in range(d):
                u = Xn[t, j] - M[k, j]
                s += u * u
            w = (cnt[k] / (cnt[k] + 1.0)) * s
            if w < bw:
                bw = w
                best = k
        out[t] = best


@njit(cache=True)
def add_points(X, idx, lab, A, cnt, labels, touched, sign):
    d = X.shape[1]
    for t in range(idx.shape[0]):
        i = idx[t]
        k = lab[t]
        for j in range(d):
            A[k, j] += sign * X[i, j]
        cnt[k] += sign
        labels[i] = k
        touched[k] = 1


def nearest(Mf, Xn):
    index = faiss.IndexFlatL2(Mf.shape[1])
    index.add(np.ascontiguousarray(Mf, dtype=np.float32))
    _, I = index.search(Xn, 1)
    return I[:, 0].astype(np.int32)


class State:
    def __init__(self, n_total, labels, A, cnt, M, neigh, lbk=None, lb=None):
        self.labels = labels.copy()
        self.A, self.cnt, self.M = A.copy(), cnt.copy(), M.copy()
        self.neigh = neigh
        self.lbk = lbk.copy() if lbk is not None else np.zeros((n_total, K), np.float32)
        self.lb = lb.copy() if lb is not None else np.zeros(n_total)
        self.Lam = np.zeros(M.shape[0])
        self.Mcp = M.copy()
        self.moved = np.zeros(n_total, np.int8)


def cert_pass(X, S, cur_idx, rng, lloyd, batch):
    """一轮带证书的精修（顺序 Hartigan 或批量 Lloyd），只作用于已插入的点 cur_idx。返回距离计算量（坐标数）。"""
    R._transfer(S.lbk, S.lb, S.labels, S.neigh, S.Lam)
    S.Lam[:] = 0.0
    S.Mcp = S.M.copy()
    order = cur_idx[rng.permutation(cur_idx.shape[0])].astype(np.int64)
    n = order.shape[0]
    tgt = np.empty(T, np.int64)
    pt = np.empty(T, np.int64)
    ps = np.empty(T, np.int64)
    dm = np.empty(T, np.float64)
    dims = 0.0
    moves = 0
    tg_all = np.empty(n, np.int64) if batch else None
    for b0 in range(0, n, T):
        b1 = min(n, b0 + T)
        R.decide(X, S.labels, S.M, S.cnt, S.neigh, order, b0, b1, tgt, True, S.lb, S.lbk, S.Lam, lloyd, pt, ps, dm)
        dims += float(dm[:b1 - b0].sum())
        if batch:
            tg_all[b0:b1] = tgt[:b1 - b0]
        else:
            m_, _r = R.commit(X, S.labels, S.A, S.M, S.cnt, order, b0, b1, tgt, S.lb, S.Lam, S.Mcp, lloyd, S.moved)
            moves += m_
    if batch:
        touched = np.zeros(S.M.shape[0], np.int8)
        moves = R.apply_batch(X, S.labels, S.A, S.cnt, order, tg_all, S.lb, S.moved, touched)
        R.recompute_touched(S.A, S.cnt, S.M, S.Mcp, S.Lam, touched)
    return dims, moves


def insert_update(X, S, new_idx, lab):
    touched = np.zeros(S.M.shape[0], np.int8)
    add_points(X, new_idx, lab, S.A, S.cnt, S.labels, touched, 1)
    tk = np.where(touched)[0]
    S.M[tk] = S.A[tk] / S.cnt[tk][:, None]
    S.Lam[tk] = np.linalg.norm(S.M[tk] - S.Mcp[tk], axis=1)   # 相对本检查点的净位移（含插入带来的位移）
    S.lb[new_idx] = 0.0                                         # 新点没有证书
    return tk


def evaluate(S, X, cur_idx, Q, gt10, pq_path=None):
    """等扫描评测（与 eval_scan 相同的粗 nprobe 网格与线性插值）。pq_path 给出时另存逐查询的 recall@10（nq × len(BUDGETS)，
    与均值曲线用同一插值权重），供种子 × 查询两层自助法与等价检验使用（只在最后一批与重建时保存）。"""
    base = X[cur_idx]
    lab = S.labels[cur_idx]
    index, quant, sizes, _ = build_ivf(S.M, base, lab)
    n = base.shape[0]
    nps = [p for p in NPROBES if p <= sizes.shape[0]]
    _, L = quant.search(Q, max(nps))
    frac, rec, p99, hits = [], [], [], []
    for npr in nps:
        index.nprobe = npr
        _, I = index.search(Q, 10)
        h = np.array([len(np.intersect1d(I[q], gt10[q])) for q in range(Q.shape[0])], np.float64)
        hits.append(h)
        rec.append(float(h.mean() / 10.0))
        sc = sizes[L[:, :npr]].sum(1)
        frac.append(float(sc.mean() / n))
        p99.append(float(np.percentile(sc, 99)))
    out = dict(cv=float(sizes.std() / sizes.mean()), max_list=int(sizes.max()), empty=int((sizes == 0).sum()))
    for b in BUDGETS:
        out[f"recall@scan{b}"] = interp(frac, rec, b)
        out[f"scan_p99@scan{b}"] = interp(frac, p99, b)
    out["recall@nprobe40"] = rec[nps.index(48)] if 48 in nps else float("nan")
    if pq_path:
        H = np.stack(hits, 1)
        pqb = np.full((Q.shape[0], len(BUDGETS)), np.nan)
        for bi, b in enumerate(BUDGETS):
            if frac[0] <= b <= frac[-1]:
                j = min(max(int(np.searchsorted(frac, b)) - 1, 0), len(frac) - 2)
                w = (b - frac[j]) / (frac[j + 1] - frac[j]) if frac[j + 1] > frac[j] else 0.0
                pqb[:, bi] = ((1 - w) * H[:, j] + w * H[:, j + 1]) / 10.0
        np.save(pq_path, pqb.astype(np.float32))
        out["pqb"] = os.path.basename(pq_path)
    return out


def build_index(X, idx, c, seed, od, tag, passes):
    """在 X[idx] 上从头建索引：C++ 分层 SuperKMeans → 精确最近质心分配 → passes 轮 CertRefine。
    返回（状态，候选图，建索引总时间，其中分层版时间）。状态以原始下标为准（未包含的点标签为 -1）。"""
    n = X.shape[0]
    sub = os.path.join(od, f"{tag}.fvecs")
    write_fvecs(sub, X[idx])
    env = dict(os.environ, SKM_HIER="1", SKM_SAVE_LABELS="1", SKM_SEED=str(seed))
    pref = os.path.join(od, f"{tag}_hier")
    with open(pref + ".log", "w") as fh:
        subprocess.run([os.environ["SCDKM_BIN"], sub, str(c), "10", "0", "0", pref], env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
    os.remove(sub)
    hier_s = cpp_train_s(pref + ".log")
    M0 = read_fvecs(pref + ".centroids.fvecs").astype(np.float64)
    Xb = np.ascontiguousarray(X[idx])
    t0 = time.perf_counter()
    neigh = R.build_neigh(M0, K)
    lab_b = nearest(M0, Xb)
    labels = np.full(n, -1, np.int32)
    labels[idx] = lab_b
    A, cnt, M = R.init_state(Xb, lab_b, c)
    S0 = State(n, labels, A, cnt, M, neigh)
    rng = np.random.default_rng(seed)
    cur = np.sort(idx).astype(np.int64)
    for _ in range(passes):
        cert_pass(X, S0, cur, rng, lloyd=False, batch=False)
    return S0, neigh, time.perf_counter() - t0 + hier_s, hier_s


def workload(X, kind, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        perm = rng.permutation(n)
        m = n // 10
        return perm[:n - m], perm[n - m:], 10
    if kind == "clustered":
        seeds = X[rng.choice(n, 20, replace=False)]
        idx = faiss.IndexFlatL2(X.shape[1])
        idx.add(np.ascontiguousarray(seeds))
        D, _ = idx.search(X, 1)
        m = n // 10
        ins = np.argsort(D[:, 0], kind="stable")[:m]
        mask = np.ones(n, bool)
        mask[ins] = False
        return np.where(mask)[0], rng.permutation(ins), 10
    if kind == "drift":
        smp = X[rng.choice(n, min(n, 100_000), replace=False)].astype(np.float64)
        mu = smp.mean(0)
        _, _, vt = np.linalg.svd(smp - mu, full_matrices=False)
        proj = (X.astype(np.float64) - mu) @ vt[0]
        order = np.argsort(proj, kind="stable")
        m = int(0.3 * n)
        return order[:n - m], order[n - m:], 6
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--workload", required=True, choices=["uniform", "clustered", "drift"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--build-passes", type=int, default=5)
    ap.add_argument("--control", action="store_true",
                    help="no-insertion control: run as many certified passes (CL and CH, same seed) on the original index "
                         "as there are insertion batches, without inserting anything; writes <workload>_s<seed>_control.json. "
                         "Separates repair gains from the continued convergence of the original partition.")
    a = ap.parse_args()
    faiss.omp_set_num_threads(20)
    X = read_fvecs(DATASETS[a.ds][0])
    n, d = X.shape
    Q = read_fvecs(DATASETS[a.ds][1], 1000)
    build_idx, ins_idx, nb = workload(X, a.workload, a.seed)
    nb0 = build_idx.shape[0]
    c = int(round(4 * np.sqrt(nb0)))
    od = os.path.join(res_dir(a.ds), "maint")
    os.makedirs(od, exist_ok=True)
    tag = f"{a.workload}_s{a.seed}"
    print(f"{a.ds} {a.workload}: build {nb0}, insert {ins_idx.shape[0]} in {nb} batches, c={c}", flush=True)
    if a.control:
        return control(X, Q, build_idx, nb, c, a, od, tag)

    # ---- 建索引：C++ 分层版 → 精确最近质心分配 → CertRefine（与主实验同一流程）
    S0, neigh, t_build, hier_s = build_index(X, build_idx, c, a.seed, od, tag + "_build", a.build_passes)
    cur = np.sort(build_idx).astype(np.int64)
    rng = np.random.default_rng(a.seed)
    print(f"built index in {t_build:.1f}s (hier {hier_s:.1f}s)", flush=True)

    S = {nm: State(n, S0.labels, S0.A, S0.cnt, S0.M, neigh, S0.lbk, S0.lb) for nm in ("frozen", "local_lloyd", "cert_lloyd", "cert_hart")}
    for nm in ("cert_lloyd", "cert_hart"):
        S[nm].Lam = S0.Lam.copy()
        S[nm].Mcp = S0.Mcp.copy()
    gt_idx = faiss.IndexFlatL2(d)
    gt_idx.add(np.ascontiguousarray(X[cur]))
    _, g = gt_idx.search(Q, 10)          # g 是在 X[cur] 中的位置下标，与 evaluate 里 IVF 的内部编号一致
    hist = [dict(batch=0, n=int(cur.shape[0]), built=evaluate(S["frozen"], X, cur, Q, g))]
    print("built index:", {k: round(v, 4) for k, v in hist[0]["built"].items() if k.startswith("recall@scan0.01") or k == "cv"}, flush=True)
    full_pass = float((K + 1) * d)
    bsz = int(np.ceil(ins_idx.shape[0] / nb))
    for b in range(nb):
        new = ins_idx[b * bsz:(b + 1) * bsz].astype(np.int64)
        Xn = np.ascontiguousarray(X[new])
        cur = np.sort(np.concatenate([cur, new]))
        row = dict(batch=b + 1, n=int(cur.shape[0]), inserted=int(new.shape[0]))
        # frozen
        s = S["frozen"]
        t0 = time.perf_counter()
        lab = nearest(s.M, Xn)
        add_points(X, new, lab, s.A, s.cnt, s.labels, np.zeros(c, np.int8), 1)
        row["frozen_cost"] = dict(s=time.perf_counter() - t0, dims_per_point=0.0)
        # local_lloyd
        s = S["local_lloyd"]
        t0 = time.perf_counter()
        lab = nearest(s.M, Xn)
        tk = insert_update(X, s, new, lab)
        mask = np.zeros(c, bool)
        mask[tk] = True
        pts = cur[mask[s.labels[cur]]]
        out = np.empty(pts.shape[0], np.int32)
        cand_nearest(X, pts, s.labels, s.M, neigh, out)
        ch = out != s.labels[pts]
        touched = np.zeros(c, np.int8)
        add_points(X, pts[ch], s.labels[pts[ch]], s.A, s.cnt, s.labels.copy(), touched, -1)
        add_points(X, pts[ch], out[ch], s.A, s.cnt, s.labels, touched, 1)
        tk2 = np.where(touched)[0]
        s.M[tk2] = s.A[tk2] / np.maximum(s.cnt[tk2], 1)[:, None]
        row["local_lloyd_cost"] = dict(s=time.perf_counter() - t0, dims_per_point=pts.shape[0] * full_pass / cur.shape[0],
                                       points_touched_pct=100.0 * pts.shape[0] / cur.shape[0])
        # cert_lloyd / cert_hart
        for nm, lloyd in (("cert_lloyd", True), ("cert_hart", False)):
            s = S[nm]
            t0 = time.perf_counter()
            lab = nearest(s.M, Xn)
            if not lloyd:
                lab2 = np.empty(new.shape[0], np.int32)
                hartigan_pick(Xn, lab, s.M, s.cnt, neigh, lab2)
                lab = lab2
            insert_update(X, s, new, lab)
            dims, moves = cert_pass(X, s, cur, rng, lloyd=lloyd, batch=lloyd)
            row[nm + "_cost"] = dict(s=time.perf_counter() - t0, dims_per_point=dims / cur.shape[0], moves=int(moves))
        gt_idx = faiss.IndexFlatL2(d)
        gt_idx.add(np.ascontiguousarray(X[cur]))
        _, g = gt_idx.search(Q, 10)
        for nm in S:
            row[nm] = evaluate(S[nm], X, cur, Q, g, os.path.join(od, f"{tag}_{nm}.pqb.npy") if b == nb - 1 else None)
        row["full_pass_dims_per_point"] = full_pass
        print(b + 1, " ".join("%s r=%.4f max=%d c%%=%.1f" % (nm, row[nm]["recall@scan0.01"], row[nm]["max_list"],
                                                         100 * row[nm + "_cost"]["dims_per_point"] / full_pass) for nm in S), flush=True)
        hist.append(row)
    # ---- 参照：在全部插入之后的数据上从头重建（同样的 c 与流程），衡量维护离"重建"有多远、省下多少代价
    SR, _, t_rebuild, _ = build_index(X, cur, c, a.seed, od, tag + "_rebuild", a.build_passes)
    rebuild = evaluate(SR, X, cur, Q, g, os.path.join(od, f"{tag}_rebuild.pqb.npy"))
    rebuild["build_s"] = t_rebuild
    print("rebuild:", {k: round(v, 4) for k, v in rebuild.items() if k in ("recall@scan0.01", "max_list", "build_s")}, flush=True)
    maint_s = {nm: float(sum(h[nm + "_cost"]["s"] for h in hist[1:])) for nm in S}
    dump_json(dict(ds=a.ds, workload=a.workload, seed=a.seed, c=c, K=K, n_build=int(nb0), n_insert=int(ins_idx.shape[0]),
                   batches=nb, build_s=t_build, rebuild=rebuild, maint_total_s=maint_s, hist=hist), os.path.join(od, f"{tag}.json"))


def control(X, Q, build_idx, nb, c, a, od, tag):
    """No-insertion control: the same index, the same seed, no inserted points; nb certified passes over all resident
    points (CL and CH kept separately), each evaluated on the original collection. Answers how much of the recall lead of
    the certified strategies over the frozen index is continued convergence of the original partition."""
    n, d = X.shape
    S0, neigh, t_build, hier_s = build_index(X, build_idx, c, a.seed, od, tag + "_control_build", a.build_passes)
    cur = np.sort(build_idx).astype(np.int64)
    rng = np.random.default_rng(a.seed)
    S = {nm: State(n, S0.labels, S0.A, S0.cnt, S0.M, neigh, S0.lbk, S0.lb) for nm in ("cert_lloyd", "cert_hart")}
    for nm in S:
        S[nm].Lam = S0.Lam.copy()
        S[nm].Mcp = S0.Mcp.copy()
    gt_idx = faiss.IndexFlatL2(d)
    gt_idx.add(np.ascontiguousarray(X[cur]))
    _, g = gt_idx.search(Q, 10)
    built = evaluate(S["cert_hart"], X, cur, Q, g, os.path.join(od, f"{tag}_control_built.pqb.npy"))
    hist = [dict(batch=0, n=int(cur.shape[0]), built=built)]
    print("built index:", {k: round(v, 4) for k, v in built.items() if k.startswith("recall@scan0.01") or k == "cv"}, flush=True)
    for b in range(nb):
        row = dict(batch=b + 1, n=int(cur.shape[0]), inserted=0)
        for nm, lloyd in (("cert_lloyd", True), ("cert_hart", False)):
            s = S[nm]
            t0 = time.perf_counter()
            dims, moves = cert_pass(X, s, cur, rng, lloyd=lloyd, batch=lloyd)
            row[nm + "_cost"] = dict(s=time.perf_counter() - t0, dims_per_point=dims / cur.shape[0], moves=int(moves))
            row[nm] = evaluate(s, X, cur, Q, g, os.path.join(od, f"{tag}_control_{nm}.pqb.npy") if b == nb - 1 else None)
        row["full_pass_dims_per_point"] = float((K + 1) * d)
        print(b + 1, " ".join("%s r=%.4f moves=%d" % (nm, row[nm]["recall@scan0.01"], row[nm + "_cost"]["moves"]) for nm in S), flush=True)
        hist.append(row)
    dump_json(dict(ds=a.ds, workload=a.workload, seed=a.seed, c=c, K=K, n_build=int(cur.shape[0]), n_insert=0, batches=nb,
                   control=True, build_s=t_build, hist=hist), os.path.join(od, f"{tag}_control.json"))


if __name__ == "__main__":
    main()
