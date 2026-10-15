#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Result-side consistency checks of the paper against the run logs and result files in experiments/results.

Checks that do not depend on make_assets.py (which the regeneration check covers):
  1. dataset statistics of Table 1 (n, d, c, number of queries) against the result files;
  2. seed coverage: every main-table configuration has the five seeds on every collection, MS MARCO likewise;
  3. log integrity: every run log finished (train_ms or a refine.json history), no error traces, no NaN recall;
  4. the sweeps and side studies the paper describes exist with the stated settings and seeds;
  5. the superseded SKM-default runs listed in the supplement are kept apart from the reported ones.

Usage:  python checks/check_results.py
"""
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "experiments", "results")
DS8 = ["sift", "glove200", "yahoo", "gist", "mxbai", "dbpedia", "wiki", "deep10m"]
SEEDS = [42, 1, 2, 3, 4]
fails, notes = [], []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------- 1. Table 1
TABLE1 = {  # as printed in Table 1 of the paper: n, d, c, n_q
    "sift": (1000000, 128, 4000, 2000), "glove200": (1183514, 200, 4352, 2000), "yahoo": (677305, 384, 3292, 1000),
    "gist": (1000000, 960, 4000, 1000), "mxbai": (769382, 1024, 3509, 1000), "dbpedia": (229772, 1536, 1917, 1000),
    "wiki": (260372, 3072, 2041, 1000), "deep10m": (9990000, 96, 12649, 2000), "msmarco": (8841823, 384, 11894, 6980)}
for ds, (n, d, c, nq) in TABLE1.items():
    r = jload(os.path.join(RES, ds, "cr_s42.refine.json"))
    if not r:
        check(False, f"Table 1 {ds}: cr_s42.refine.json missing")
        continue
    hits = glob.glob(os.path.join(RES, ds, "faiss_s42*.hits.npy"))
    q = None
    if hits:
        import numpy as np
        q = int(np.load(hits[0], mmap_mode="r").shape[0])
    if ds == "msmarco":
        meta = jload(os.path.join(HERE, "..", "experiments", "data_msmarco", "msmarco_meta.json"))
        q = meta["nq"] if meta else q
    ok = (r["n"], r["d"], r["c"]) == (n, d, c) and (q is None or q == nq)
    check(ok, f"Table 1 {ds}: n={r['n']} d={r['d']} c={r['c']} queries={q} (paper {n}, {d}, {c}, {nq})")
    expect_c = round(4 * math.sqrt(1e7 if ds == "deep10m" else n))
    check(c == expect_c, f"Table 1 {ds}: c = round(4*sqrt(n)) -> {expect_c}" + (" (nominal n = 10^7)" if ds == "deep10m" else ""))

# ---------------------------------------------------------------- 2. seed coverage of the main comparison
FLAT = ["hier", "faiss_2lc", "hsl_cpp", "hor1", "hor2", "hor", "warm1", "warm2", "warm3", "skm_default", "faiss_ivf", "faiss", "lloyd"]
REFINE = [("cr_labels", 5), ("cr", 1), ("cr", 5), ("hslc_labels", 5), ("hslc", 5), ("hsl", 5)]
for ds in DS8:
    miss = []
    for t in FLAT:
        miss += [f"{t}_s{s}" for s in SEEDS if not os.path.exists(os.path.join(RES, ds, f"{t}_s{s}.scan.json"))]
    for t, e in REFINE:
        miss += [f"{t}_s{s}_ep{e}" for s in SEEDS if not os.path.exists(os.path.join(RES, ds, f"{t}_s{s}_ep{e}.scan.json"))]
    check(not miss, f"main comparison {ds}: {len(FLAT) + len(REFINE)} configurations x 5 seeds" + (f", missing {miss}" if miss else ""))
ms_tags = ["hier", "faiss_2lc", "cr_labels_ep5", "cr_ep1", "cr_ep5", "hslc_labels_ep5", "hslc_ep5", "hsl_cpp", "hor1", "hor2", "hor",
           "warm1", "warm2", "warm3", "skm_default", "faiss_ivf", "faiss"]
miss = []
for t in ms_tags:
    base, _, ep = t.partition("_ep")
    for s in SEEDS:
        name = f"{base}_s{s}_ep{ep}" if ep else f"{t}_s{s}"
        if not glob.glob(os.path.join(RES, "msmarco", name + ".ir.json")):
            miss.append(name)
check(not miss, f"MS MARCO: {len(ms_tags)} rows of Table 6 x 5 seeds" + (f", missing {miss}" if miss else ""))
for t, k in (("sl", 5), ("faiss_sampled", 3)):
    have = [s for s in SEEDS if os.path.exists(os.path.join(RES, "msmarco", f"{t}_s{s}" + ("_ep5" if t == "sl" else "") + ".ir.json"))]
    check(len(have) == k, f"MS MARCO untested rows: {t} has {len(have)} seeds (Table 6 note: FAISS-sampled 3 seeds, SL-5 five)")

# ---------------------------------------------------------------- 3. log integrity
bad_pat = re.compile(r"Traceback|Segmentation fault|Killed|core dumped|out of memory|MemoryError|\bERROR\b", re.I)
nlogs, bad, unfinished = 0, [], []
for ds in os.listdir(RES):
    d = os.path.join(RES, ds)
    if not os.path.isdir(d):
        continue
    for p in glob.glob(os.path.join(d, "*.log")):
        nlogs += 1
        s = open(p, encoding="utf-8", errors="replace").read()
        if bad_pat.search(s):
            bad.append(os.path.relpath(p, RES))
        if "train_ms" not in s and "total_s" not in s and "_superseded" not in p:
            unfinished.append(os.path.relpath(p, RES))
check(not bad, f"run logs: {nlogs} logs, none with an error trace" + (f", see {bad[:8]}" if bad else ""))
check(not unfinished, f"run logs: every log records its training time" + (f", not: {unfinished[:8]}" if unfinished else ""))
nan = []
nscan = 0
for p in glob.glob(os.path.join(RES, "*", "*.scan.json")):
    nscan += 1
    j = jload(p)
    v = j.get("recall@scan0.01") if j else None
    if v is None or (isinstance(v, float) and math.isnan(v)):
        nan.append(os.path.relpath(p, RES))
check(not nan, f"evaluations: {nscan} scan.json files, recall at the 1% budget present and finite" + (f", not: {nan[:8]}" if nan else ""))
short = []
for p in glob.glob(os.path.join(RES, "*", "*.refine.json")):
    j = jload(p)
    if not j or "hist" not in j or not j["hist"]:
        short.append(os.path.relpath(p, RES))
check(not short, "refinement runs: every refine.json carries its per-pass history")

# ---------------------------------------------------------------- 4. sweeps and side studies
def seeds_of(ds, pattern):
    return sorted({int(m.group(1)) for p in glob.glob(os.path.join(RES, ds, pattern))
                   for m in [re.search(r"_s(\d+)[._]", os.path.basename(p))] if m})


for K in (16, 32, 128, 256):
    got = {ds: seeds_of(ds, f"cr_K{K}_s*_ep5.scan.json") for ds in DS8 if ds != "deep10m"}
    check(all(v == [1, 2, 42] for v in got.values()), f"K sweep K={K}: seeds 42, 1, 2 on the seven collections other than Deep-10M")
for c in (1000, 2000, 8000):
    got = seeds_of(f"sift_c{c}", "cr_s*_ep5.scan.json")
    check(got == [1, 2, 42], f"list-count sweep c={c} on SIFT: seeds {got} (paper: 3 seeds)")
for ds in ("sift", "gist"):
    thr = [t for t in (1, 2, 5, 10, 20) if os.path.exists(os.path.join(RES, ds, f"cr_thr{t}_s42.refine.json"))]
    check(thr == [1, 2, 5, 10, 20], f"thread sweep {ds}: threads {thr}")
bs = sorted({int(m.group(1)) for p in glob.glob(os.path.join(RES, "*", "cr_B*_s42.refine.json"))
             for m in [re.search(r"cr_B(\d+)_", os.path.basename(p))] if m})
check(bs and bs[0] == 256 and bs[-1] == 16384, f"batch-size sweep: B in {bs} (paper: 256 to 16,384)")
fs = sorted({os.path.basename(p).split("_s")[0] for p in glob.glob(os.path.join(RES, "sift", "faiss_it*_s*.scan.json"))})
want = {f"faiss_it{i}_m{m}" for i in (5, 10, 25) for m in (39, 64, 128)} | {"faiss_it5_full", "faiss_it10_full"}
check(set(fs) >= want and os.path.exists(os.path.join(RES, "sift", "faiss_s42.scan.json")),
      f"FAISS sweep: {len(fs)} configurations on SIFT plus FAISS-full as the 25-iteration full-data point "
      "(5/10/25 iterations x 39/64/128 points per centroid or the whole collection)")
fsd = {ds: seeds_of(ds, "faiss_it10_m64_s*.scan.json") for ds in DS8}
check(all(v == [1, 2, 42] for d, v in fsd.items() if d != "deep10m") and fsd["deep10m"] == [42],
      f"FAISS sweep seeds: 3 on the seven smaller collections, 1 on Deep-10M ({fsd})")
for ds in ("sift", "yahoo"):
    for w in ("uniform", "clustered", "drift"):
        have = [s for s in (42, 1, 2) if os.path.exists(os.path.join(RES, ds, "maint", f"{w}_s{s}.json"))]
        ctrl = os.path.exists(os.path.join(RES, ds, "maint", f"{w}_s42_control.json"))
        check(have == [42, 1, 2] and ctrl, f"maintenance {ds}/{w}: seeds {have}, no-insertion control {ctrl}")
xc = [ds for ds in DS8 if os.path.exists(os.path.join(RES, ds, "xcr_s42.refine.json")) or glob.glob(os.path.join(RES, ds, "xcr_s42*"))]
check(len(xc) == 8, f"C++ port cross-check: {len(xc)} of 8 collections")
qps = [ds for ds in DS8 if glob.glob(os.path.join(RES, ds, "*_s42.qps.json"))]
check(len(qps) == 8, f"query throughput: qps.json on {len(qps)} of 8 collections")
pm = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RES, "peakmem", "*.txt")))
check(any(p.startswith("deep10m_") for p in pm) and any(p.startswith("msmarco_") for p in pm), f"peak memory: {len(pm)} GNU time reports")

# ---------------------------------------------------------------- 5. superseded SKM-default runs
sup = sorted(os.path.relpath(p, RES) for p in glob.glob(os.path.join(RES, "*", "_superseded", "skm_default_s*.log")))
print(f"INFO superseded SKM-default logs kept locally: {len(sup)}")
newcfg = [p for p in glob.glob(os.path.join(RES, "*", "skm_default_s*.log"))
          if re.search(r"nq=1000", open(p, errors="replace").read().splitlines()[0] if open(p, errors="replace").read() else "")]
allskm = glob.glob(os.path.join(RES, "*", "skm_default_s*.log"))
check(len(newcfg) == len(allskm), f"SKM-default: all {len(allskm)} reported runs stop early on 1,000 sampled queries (config nq=1000)")

print()
print(f"== {len(fails)} check(s) failed")
sys.exit(1 if fails else 0)
