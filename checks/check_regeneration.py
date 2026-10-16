#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate every number, table and figure of the paper from the run logs into a temporary folder.

1. runs experiments/analysis/make_assets.py with PAPER_DIR set to a temporary folder (about three minutes);
2. checks that it ran without error and that no claim failed, and lists the inputs it reports as missing;
3. counts what it produced: data keys of generated/values.tex, tables generated/tab_*.tex, figures figs/*.pdf.

With --compare DIR, it also compares the output with DIR/generated and DIR/figs of a paper folder:
generated files byte by byte (ignoring the first comment line, which names the generator path) and figures pixel by
pixel (needs pdftoppm from poppler, and Pillow). Figures that the paper redraws in another layout are skipped.

Usage:  python checks/check_regeneration.py [--compare DIR] [--keep OUT_DIR]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MAKE = os.path.join(ROOT, "experiments", "analysis", "make_assets.py")
REDRAWN = {"fig_pareto.pdf", "fig_fsweep.pdf", "fig_cert.pdf"}   # Figures 3, 4 and 6: same data, one-row layout in the paper
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def body(path):
    """File content without a first line that is a comment naming the generator."""
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    if lines and re.match(r"^\s*[%#].*make_assets\.py", lines[0]):
        lines = lines[1:]
    return lines


def pixels_equal(a, b, work):
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return None
    imgs = []
    for i, pdf in enumerate((a, b)):
        base = os.path.join(work, f"cmp{i}")
        try:
            if subprocess.run(["pdftoppm", "-r", "110", "-png", "-singlefile", pdf, base], capture_output=True).returncode:
                return None
        except FileNotFoundError:
            return None
        imgs.append(Image.open(base + ".png").convert("RGB"))
    return imgs[0].size == imgs[1].size and ImageChops.difference(*imgs).getbbox() is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", help="paper folder with generated/ and figs/ to compare with")
    ap.add_argument("--keep", help="write the output here instead of a temporary folder")
    a = ap.parse_args()
    out = os.path.abspath(a.keep) if a.keep else tempfile.mkdtemp(prefix="certrefine_regen_")
    try:
        env = dict(os.environ, PAPER_DIR=out)
        r = subprocess.run([sys.executable, MAKE], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = r.stdout + r.stderr
        open(os.path.join(HERE, "make_assets_last_run.log"), "w", encoding="utf-8").write(log)
        check(r.returncode == 0, "make_assets.py ran without error (output in checks/make_assets_last_run.log)")
        check("ERROR" not in log, "make_assets.py reported no failed claim (no ERROR line)")
        print("INFO inputs reported missing (both gaps are stated in the paper):",
              [line for line in log.splitlines() if line.startswith("MISSING")])
        gen, figs = os.path.join(out, "generated"), os.path.join(out, "figs")
        vals = os.path.join(gen, "values.tex")
        nkeys = len(re.findall(r"\\defV\{", open(vals, encoding="utf-8").read())) if os.path.exists(vals) else 0
        tabs = sorted(f for f in os.listdir(gen) if f.startswith("tab_")) if os.path.isdir(gen) else []
        pdfs = sorted(f for f in os.listdir(figs) if f.endswith(".pdf")) if os.path.isdir(figs) else []
        check(nkeys > 0 and tabs and pdfs, f"produced {nkeys} data keys, {len(tabs)} tables and {len(pdfs)} figures")

        if a.compare:
            ref_gen, ref_figs = os.path.join(a.compare, "generated"), os.path.join(a.compare, "figs")
            names = sorted(os.listdir(gen))
            diff = [n for n in names if not os.path.exists(os.path.join(ref_gen, n))
                    or body(os.path.join(gen, n)) != body(os.path.join(ref_gen, n))]
            check(not diff, f"generated/: {len(names)} files identical to {ref_gen}" + (f", differing: {diff}" if diff else ""))
            work = tempfile.mkdtemp(prefix="certrefine_px_")
            for f in pdfs:
                if f in REDRAWN or not os.path.exists(os.path.join(ref_figs, f)):
                    print(f"SKIP {f}: drawn in another layout by the paper" if f in REDRAWN else f"SKIP {f}: not in {ref_figs}")
                    continue
                eq = pixels_equal(os.path.join(figs, f), os.path.join(ref_figs, f), work)
                if eq is None:
                    print(f"SKIP {f}: pdftoppm or Pillow not available")
                else:
                    check(eq, f"{f}: identical pixels")
            shutil.rmtree(work, ignore_errors=True)
    finally:
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)
    print(f"\n== {len(fails)} check(s) failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
