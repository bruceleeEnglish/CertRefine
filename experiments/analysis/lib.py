# -*- coding: utf-8 -*-
"""结果解析与统计（本地运行；输入为 experiments/results/）。

约定：
  C++/FAISS/sklearn 的一次训练 = <cfg>_s<seed>.log（含 train_ms）+ <cfg>_s<seed>.scan.json（等扫描评测）
  numba 精修的一次运行 = <tag>_s<seed>.refine.json（每轮 total_s 等）+ <tag>_s<seed>_ep<e>.scan.json
  构建时间：C++ 取 train_ms；hsl_cpp 另加同种子分层版的 train_ms（它从分层质心出发）；精修取 hist[e].total_s
  （已含分层版时间、初始分配、候选图与各轮）。
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RES_LOCAL", os.path.join(HERE, "..", "results"))
DS8 = ["sift", "glove200", "yahoo", "gist", "mxbai", "dbpedia", "wiki", "deep10m"]
DSNAME = dict(sift="SIFT", glove200="GloVe", yahoo="Yahoo", gist="GIST", mxbai="MXBAI", dbpedia="DBpedia",
              wiki="Wiki", deep10m="Deep-10M", msmarco="MS MARCO")
SEEDS = [42, 1, 2, 3, 4]
B1 = 0.01
BUDGETS = [0.0025, 0.005, 0.01, 0.02, 0.04]


def train_s(path):
    for line in open(path, errors="replace"):
        if line.startswith("train_ms,"):
            return float(line.split(",")[1]) / 1000.0
    return float("nan")


def _scan(d, name):
    p = os.path.join(d, name + ".scan.json")
    return json.load(open(p)) if os.path.exists(p) else None


def assign_s(d):
    """最终列表分配（全部向量精确分配到 c 个质心）的耗时，所有方法相同。
    优先用 time_assign.py 的实测（assign_time.json，3 次中位数）；没有时（如 c 扫描目录）用同目录 CertRefine 运行里
    同一操作（初始精确分配，init_assign_s）的中位数。"""
    ds = os.path.basename(os.path.normpath(d))
    p = os.path.join(RES, "assign_time.json")
    if os.path.exists(p):
        at = json.load(open(p))
        if ds in at:
            return float(at[ds]["assign_s"])
    vals = []
    for rj in glob.glob(os.path.join(d, "cr_s*.refine.json")):
        vals.append(json.load(open(rj))["timing"]["init_assign_s"])
    return float(np.median(vals)) if vals else float("nan")


def runs(ds, e2e=True):
    """返回 {(method, epoch): {seed: dict(time=, time_train=, scan=, name=)}}；epoch=None 表示非迭代的一次性训练。
    time 为端到端构建时间（训练 + 最终列表分配，e2e=True）；time_train 只含训练。"""
    d = os.path.join(RES, ds)
    fa = assign_s(d) if e2e else 0.0
    out = {}
    for log in glob.glob(os.path.join(d, "*_s*.log")):
        m = re.match(r"(.+)_s(\d+)\.log$", os.path.basename(log))
        if not m:
            continue
        cfg, seed = m.group(1), int(m.group(2))
        t = train_s(log)
        if cfg == "hsl_cpp":
            hl = os.path.join(d, f"hier_s{seed}.log")
            t = t + train_s(hl) if os.path.exists(hl) else float("nan")
        name = f"{cfg}_s{seed}"
        out.setdefault((cfg, None), {})[seed] = dict(time=t + fa, time_train=t, scan=_scan(d, name), name=name, dir=d)
    for rj in glob.glob(os.path.join(d, "*_s*.refine.json")):
        m = re.match(r"(.+)_s(\d+)\.refine\.json$", os.path.basename(rj))
        tag, seed = m.group(1), int(m.group(2))
        r = json.load(open(rj))
        for h in r["hist"][1:]:
            e = h["epoch"]
            name = f"{tag}_s{seed}_ep{e}"
            t = h.get("total_s", float("nan"))
            out.setdefault((tag, e), {})[seed] = dict(time=t + fa, time_train=t, scan=_scan(d, name),
                                                     name=name, dir=d, hist=h, refine=r)
    return out


def agg(entry, key="recall@scan0.01", seeds=None):
    """对某个 (method, epoch) 的各种子取均值/标准差：返回 (time_mean, time_sd, val_mean, val_sd, n_seeds)。"""
    seeds = seeds or [s for s in SEEDS if s in entry]
    ts = [entry[s]["time"] for s in seeds if s in entry]
    vs = [entry[s]["scan"][key] for s in seeds if s in entry and entry[s]["scan"] is not None]
    f = lambda a: (float(np.mean(a)) if a else float("nan"), float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"))
    return (*f(ts), *f(vs), len(vs))


def per_query_at_budget(run, b=B1):
    """逐查询 recall@10 在扫描预算 b 处的值。稠密评测（eval_scan_gpu.py）直接给出 pqb.npy；
    旧的粗网格结果则在 hits.npy 上线性插值（与均值曲线使用同一插值权重）。"""
    pq = os.path.join(run["dir"], run["name"] + ".pqb.npy")
    if os.path.exists(pq) and b in BUDGETS:
        return np.load(pq)[:, BUDGETS.index(b)].astype(np.float64)
    sc = run["scan"]
    hits = np.load(os.path.join(run["dir"], run["name"] + ".hits.npy")).astype(np.float64) / 10.0
    fr = np.asarray(sc["scan_frac"])[:hits.shape[1]]
    j = int(np.searchsorted(fr, b)) - 1
    j = min(max(j, 0), len(fr) - 2)
    w = (b - fr[j]) / (fr[j + 1] - fr[j])
    return (1 - w) * hits[:, j] + w * hits[:, j + 1]


def seed_avg_pq(entry, b=B1):
    arr = [per_query_at_budget(entry[s], b) for s in SEEDS if s in entry and entry[s]["scan"] is not None]
    return np.mean(arr, axis=0) if arr else None


def per_seed_pq(entry, b=B1, seeds=None):
    """(种子数 × 查询数) 的逐查询召回矩阵；seeds 缺省为全部可用种子。"""
    seeds = seeds or SEEDS
    arr = [per_query_at_budget(entry[s], b) for s in seeds if s in entry and entry[s]["scan"] is not None]
    return np.vstack(arr) if arr else None


def boot_diff(A, B, n_boot=2000, seed=0, delta=None):
    """两层自助法（训练种子 × 查询）：A、B 为 (种子 × 查询) 矩阵，查询相同、种子各自独立重抽。
    每次重抽：先对查询有放回抽样（两种方法用同一组查询，保持配对），再各自对种子有放回抽样，取均值之差。
    这样训练的随机性（不同种子建出的索引不同）也进入了推断，而不只是查询的随机性。
    返回 dict(diff, lo95, hi95, lo90, hi90, p, equiv)，equiv 表示 90% 区间落在 ±delta 内（TOST，α=0.05）。"""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    rng = np.random.default_rng(seed)
    nq = A.shape[1]
    qi = rng.integers(0, nq, size=(n_boot, nq))
    ma = np.stack([A[s][qi].mean(1) for s in range(A.shape[0])], axis=1)      # n_boot × 种子
    mb = np.stack([B[s][qi].mean(1) for s in range(B.shape[0])], axis=1)
    ia = rng.integers(0, A.shape[0], size=(n_boot, A.shape[0]))
    ib = rng.integers(0, B.shape[0], size=(n_boot, B.shape[0]))
    d = np.take_along_axis(ma, ia, axis=1).mean(1) - np.take_along_axis(mb, ib, axis=1).mean(1)
    obs = float(A.mean() - B.mean())
    lo95, hi95 = np.percentile(d, [2.5, 97.5])
    lo90, hi90 = np.percentile(d, [5, 95])
    p = max(2 * min(float((d <= 0).mean()), float((d >= 0).mean())), 1.0 / n_boot)
    eq = bool(delta is not None and lo90 > -delta and hi90 < delta)
    return dict(diff=obs, lo95=float(lo95), hi95=float(hi95), lo90=float(lo90), hi90=float(hi90), p=float(p), equiv=eq)


def paired_test(a, b, n_perm=10000, seed=0):
    """双侧配对随机化（符号翻转）检验：返回 (均值差, p 值)。"""
    diff = np.asarray(a) - np.asarray(b)
    obs = diff.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.shape[0]))
    null = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-15) + 1) / (n_perm + 1)
    return float(obs), float(p)


def holm(pvals):
    """Holm–Bonferroni 调整后的 p 值（保持输入顺序）。"""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj
