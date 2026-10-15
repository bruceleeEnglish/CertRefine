# -*- coding: utf-8 -*-
"""Timing-outlier rule used before the analysis (disclosed in the paper, Section "Configuration and timing").

A run is flagged for re-timing if its build time deviates by more than 15% (in either direction) from the
median of the other seeds of the same method on the same collection. Re-timing re-runs the training with
the same seed; results are deterministic given the seed, so only the time changes. The rule is applied
symmetrically to every method with at least three seeds.

Usage (local, on the pulled results):  python experiments/tools/timing_outliers.py [--tol 0.15]
Prints the flagged runs in the format of logs/retime_list.txt (collection:method:seed).

History: the first application (2026-09-23, seeds 42/1/2) flagged GIST SKM-Lloyd (seed 42), GIST batched
CDKM (seed 42), DBpedia batched CDKM (seed 2), GloVe HSL+cert (seed 1; initial assignment 3x slower than
in the other seeds) and two SKM-default runs that were later superseded by the leakage-free re-runs.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=0.15)
ap.add_argument("--min-seeds", type=int, default=3)
a = ap.parse_args()

flag = []
for ds in L.DS8:
    if not os.path.isdir(os.path.join(L.RES, ds)):
        continue
    R = L.runs(ds, e2e=False)
    for (m, e), en in sorted(R.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        if e not in (None, 5):          # iterative runs: judge the complete five-pass run only
            continue
        t = {s: v["time_train"] for s, v in en.items() if not np.isnan(v["time_train"])}
        if len(t) < a.min_seeds:
            continue
        for s, x in t.items():
            ref = float(np.median([y for s2, y in t.items() if s2 != s]))
            r = x / ref
            if r > 1 + a.tol or r < 1 / (1 + a.tol):
                flag.append((ds, m, s, x, ref, r))
for ds, m, s, x, ref, r in flag:
    print(f"{ds}:{m}:{s}   time={x:.2f}s  median(other seeds)={ref:.2f}s  ratio={r:.2f}")
print(f"# {len(flag)} run(s) outside +-{100 * a.tol:.0f}% of the median of the other seeds", file=sys.stderr)
