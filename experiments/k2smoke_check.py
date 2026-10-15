#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke test for the candidate-graph refresh branch (R = 1) before running the k2-means-style configuration on all
collections: the first pass must equal the R = infinity run of the same seed (hslc_labels_s42), every later pass must
have rebuilt the graph, at least one later pass must differ, and the run must not be absurdly slow. Exit code 0 = OK."""
import json
import os
import sys

res = os.environ.get("RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
try:
    a = json.load(open(os.path.join(res, "sift", "k2smoke_s42.refine.json")))["hist"]
    b = json.load(open(os.path.join(res, "sift", "hslc_labels_s42.refine.json")))["hist"]
    ok = (a[1]["labels_md5"] == b[1]["labels_md5"] and all(h["graph_rebuilt"] for h in a[2:])
          and any(x["labels_md5"] != y["labels_md5"] for x, y in zip(a[2:], b[2:])) and a[-1]["total_s"] < 5 * b[-1]["total_s"])
    print("K2SMOKE", "OK" if ok else "FAIL", {"pass1_equal": a[1]["labels_md5"] == b[1]["labels_md5"],
                                               "rebuilt": [h["graph_rebuilt"] for h in a[1:]],
                                               "total_s": (a[-1]["total_s"], b[-1]["total_s"])})
except Exception as e:  # noqa: BLE001
    ok = False
    print("K2SMOKE error", repr(e))
sys.exit(0 if ok else 1)
