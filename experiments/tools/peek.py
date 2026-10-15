#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速查看某数据集已有的等扫描评测结果。用法: peek.py <ds> [name_prefix ...]"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import res_dir  # noqa: E402

ds = sys.argv[1]
pref = sys.argv[2:]
od = res_dir(ds)
rows = []
for f in sorted(glob.glob(os.path.join(od, "*.scan.json"))):
    nm = os.path.basename(f)[:-len(".scan.json")]
    if pref and not any(nm.startswith(p) for p in pref):
        continue
    r = json.load(open(f))
    rows.append((nm, r["recall@scan0.01"], r["recall@scan0.005"], r["cv"], r["wcss"]))
for nm, r1, r05, cv, w in rows:
    print("%-28s r@1%%=%.4f r@0.5%%=%.4f cv=%.3f wcss=%.5g" % (nm, r1, r05, cv, w))
