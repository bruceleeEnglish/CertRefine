#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由实验结果生成论文用的全部数字（generated/values.tex）、方法名宏（generated/macros.tex）、表格（generated/tab_*.tex）与图（figs/*.pdf）。
用法（本地）: python experiments/analysis/make_assets.py
正文里的每个数字都通过 \\V{key} 引用这里写出的键；缺失的键在 PDF 中显示为红色 ??key。
缺失的结果文件不会导致异常：对应的行/键被跳过并打印 `MISSING ...`；文件到位后重跑即可自动补全。
统计口径统一（显著性优先、等价另标、每个数据集内对全部对照做 Holm），方法名全部经宏输出（RENAME_B 一键改名）。"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib as L  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# write generated/ and figs/ into $PAPER_DIR, by default paper/ of this repository
PAPER = os.environ.get("PAPER_DIR", os.path.join(HERE, "..", "..", "paper"))
GEN = os.path.join(PAPER, "generated")
FIG = os.path.join(PAPER, "figs")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- 风格：统一调色板（每个方法、每个集合各一个颜色）
# 8 个彩色在 OKLCH 中取中等亮度与彩度（同一明度/彩度带 → 整体协调），色相均匀分布并交替明暗（→ 彼此可分）；
# dataviz validate_palette.js：亮度带、彩度下限、正常视觉 ΔE 全部通过，相邻色盲 ΔE >= 7.5（组内另有标记形状作第二编码）。
BLUE, ORANGE, TEAL, PURPLE, CORAL, OLIVE, GREEN, ROSE = "#3475c4", "#c26e12", "#209993", "#754d9e", "#fb817a", "#a49c38", "#27762f", "#ab517f"
GRAY_L, GRAY_D, BROWN = "#a4a4a4", "#585858", "#8a5a3c"
PALETTE8 = [BLUE, ORANGE, TEAL, PURPLE, CORAL, OLIVE, GREEN, ROSE]     # 已验证的相邻顺序（集合用）
C_OURS = BLUE
AQUA, YELLOW, MAGENTA, VIOLET, RED = TEAL, OLIVE, ROSE, PURPLE, CORAL   # 旧颜色名映射到新调色板
CAT8 = PALETTE8
INK, INK2, MUTED, GRID, AXIS, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#ebeae5", "#c3c2b7", GRAY_L
FIGW = 6.48          # 版心宽度（英寸）：图按原尺寸排版，字号即印刷字号


def FS(h):
    """原先按 7.2 英寸宽设计的图高，按比例换到版心宽度。"""
    return (FIGW, h * FIGW / 7.2)


# 字体：Roboto（与示意图的 Roboto Slab 同族）。TeX Live 的 OTF 已转为 TrueType（fonts/otf2ttf.py），PDF 里以 Type 42 嵌入。
from matplotlib import font_manager as _fm
for _f in ("Roboto-Regular", "Roboto-Medium", "Roboto-Bold", "Roboto-Italic"):
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", _f + ".ttf")
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)
plt.rcParams.update({
    "font.family": "Roboto", "font.size": 7.5, "axes.titlesize": 7.8, "axes.titleweight": "medium",
    "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 7,
    "mathtext.fontset": "custom", "mathtext.rm": "Roboto", "mathtext.it": "Roboto:italic", "mathtext.bf": "Roboto:bold",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 2.2, "lines.linewidth": 1.3,
    "lines.markersize": 4.5, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def top_legend(fig, handles, labels, ncol):
    """图例放在所有子图上方（不压坐标轴），并给它留出空间。"""
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.8, w_pad=0.6)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handletextpad=0.4)


DS = L.DS8
V = {}          # 正文数字
SUMMARY = {}    # 机器可读汇总
MISSING = []    # 缺失的结果（打印 MISSING ...，不抛异常）


def missing(what):
    if what not in MISSING:
        MISSING.append(what)
        print("MISSING", what)


def fmt(x, nd=1):
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "--"
    s = f"{x:.{nd}f}"
    if s.startswith("-"):
        # 负数用真正的减号（正文与数学模式都可用）；"-0.00" 这类负零去掉符号
        s = s[1:] if float(s) == 0 else r"\ensuremath{-}" + s[1:]
    return s


def setv(key, val, nd=None):
    if isinstance(val, (bool, np.bool_)):
        V[key] = "yes" if val else "no"
    elif isinstance(val, (int, np.integer)):
        V[key] = str(int(val))
    elif isinstance(val, (float, np.floating)):
        V[key] = fmt(float(val), 1 if nd is None else nd)
    else:
        V[key] = str(val)


def write_gen(name, text):
    open(os.path.join(GEN, name), "w", encoding="utf-8").write(text)


def isnum(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


# ---------------------------------------------------------------- 0. 方法名的唯一来源（改名开关）
RENAME_B = True   # False -> 带证书的批量 Lloyd 沿用旧的基线名（HSL+cert-L / HSL+cert）
NAMES = {  # (代码 tag, epoch) -> (宏名, 打印名, 方法族, 分组)
    ("hier", None):        ("hierskm",  "Hier-SKM",       "Hier-SKM",     "hier"),
    ("faiss_2lc", None):   ("faisstwol", "FAISS-2L",      "FAISS-2L",     "hier"),      # faiss.contrib.clustering.two_level_clustering
    ("cr_labels", 5):      ("oursHL",   "CertRefine-H-L", "CertRefine-H", "ours"),
    ("cr", 1):             ("oursHone", "CertRefine-H-1", "CertRefine-H", "ours"),
    ("cr", 5):             ("oursHfive", "CertRefine-H-5", "CertRefine-H", "ours"),
    ("hslc_labels", 5):    ("oursBL",   "CertRefine-B-L" if RENAME_B else "HSL+cert-L", "CertRefine-B", "ours"),
    ("hslc", 5):           ("oursBfive", "CertRefine-B-5" if RENAME_B else "HSL+cert",   "CertRefine-B", "ours"),
    ("hsl", 5):            ("hslnocert", "HSL (no cert)",  "HSL",          "refine"),   # 只进时间表与补充材料
    ("hsl_cpp", None):     ("hslskm",   "HSL-SKM",        "HSL-SKM",      "refine"),
    ("hor1", None):        ("horone",   "HOR-1",          "HOR",          "refine"),
    ("hor2", None):        ("hortwo",   "HOR-2",          "HOR",          "refine"),
    ("hor", None):         ("horfive",  "HOR-5",          "HOR",          "refine"),
    ("warm1", None):       ("wsone",    "Lloyd-WS-1",     "Lloyd-WS",     "refine"),
    ("warm2", None):       ("wstwo",    "Lloyd-WS-2",     "Lloyd-WS",     "refine"),
    ("warm3", None):       ("wsthree",  "Lloyd-WS-3",     "Lloyd-WS",     "refine"),
    ("skm_default", None): ("skmdef",   "SKM-default",    "SKM-default",  "sampled"),
    ("faiss_ivf", None):   ("faissivf", "FAISS-IVF",      "FAISS",        "sampled"),
    ("faiss", None):       ("faissfull", "FAISS-full",    "FAISS",        "full"),
    ("lloyd", None):       ("skmlloyd", "SKM-Lloyd",      "SKM-Lloyd",    "full"),
    ("sl", 5):             ("slfive",   "SL-5",           "CertRefine-H", "supp"),      # 顺序 Lloyd 判据（消融），MS MARCO 表中列出但不检验
    ("sk_minibatch", None): ("minibatch", "MiniBatch",    "MiniBatch",    "supp"),      # 补充材料，见 §6
    ("k2style", 5):        ("ktwostyle", "CertRefine-B-L ($R{=}1$)", "CertRefine-B", "supp"),
}
BY_MACRO = {v[0]: k for k, v in NAMES.items()}          # 宏名 -> (tag, epoch)
PRINTED = {v[0]: v[1] for v in NAMES.values()}           # 宏名 -> 打印名（图例、日志）
FAMILY_OF = {v[0]: v[2] for v in NAMES.values()}
GROUP_OF = {v[0]: v[3] for v in NAMES.values()}
GROUP_TITLE = {"hier": "Hierarchical start", "ours": "Certified refinement (this paper)",
               "refine": "Other refinements of the same partition", "sampled": "Sampled", "full": "Full data"}
MAIN_MACROS = ["hierskm", "faisstwol", "oursHL", "oursHone", "oursHfive", "oursBL", "oursBfive",
               "hslskm", "horone", "hortwo", "horfive", "wsone", "wstwo", "wsthree", "skmdef", "faissivf", "faissfull", "skmlloyd"]
MAIN = [(BY_MACRO[m][0], BY_MACRO[m][1], m, GROUP_OF[m]) for m in MAIN_MACROS]   # (tag, epoch, 宏名, 分组)
HSLROW = ("hsl", 5, "hslnocert", "refine")
# 旧的短名（正文里 sp.*/ds.* 等键沿用）-> (tag, epoch)
SHORT = {"cr5": ("cr", 5), "cr1": ("cr", 1), "crl": ("cr_labels", 5), "bl": ("hslc_labels", 5), "b5": ("hslc", 5), "hsl": ("hsl", 5),
         "hier": ("hier", None), "fivf": ("faiss_ivf", None), "ffull": ("faiss", None), "lloyd": ("lloyd", None), "hor": ("hor", None),
         "hor1": ("hor1", None), "hor2": ("hor2", None), "hslskm": ("hsl_cpp", None), "ws1": ("warm1", None), "ws2": ("warm2", None),
         "ws3": ("warm3", None), "skmdef": ("skm_default", None), "tlc": ("faiss_2lc", None)}
MACRO_OF_SHORT = {k: NAMES[v][0] for k, v in SHORT.items()}
MARKER = {"hierskm": "o", "faisstwol": "8", "oursHone": "o", "oursHfive": "s", "oursHL": "d", "oursBL": "<", "oursBfive": "v", "hslnocert": "^",
          "hslskm": ">", "horone": "D", "hortwo": "D", "horfive": "D", "wsone": "h", "wstwo": "h", "wsthree": "h",
          "skmdef": "P", "faissivf": "*", "faissfull": "*", "skmlloyd": "X", "minibatch": "p", "slfive": "o"}
# 方法族（Pareto 图）：同一族的工作点连成一条线；所有族同样的标记大小与线宽（本文方法只保留蓝色，不做视觉强调）
FAM_STYLE = {"CertRefine-H": (BLUE, "-"), "CertRefine-B": (PURPLE, "-"), "HSL-SKM": (ORANGE, "-"), "HOR": (CORAL, "-"),
             "Lloyd-WS": (OLIVE, "-"), "SKM-default": (ROSE, "-"), "FAISS": (TEAL, "-"), "SKM-Lloyd": (GREEN, "-"),
             "Hier-SKM": (GRAY_L, "-"), "FAISS-2L": (GRAY_D, "-"), "MiniBatch": (BROWN, "-")}
HOLLOW = set()      # 颜色已各不相同：标记一律实心
# 单方法曲线（扫描预算图、吞吐图）：颜色同上；同一族的第二个工作点用虚线
M_STYLE = {"oursHfive": (BLUE, "-"), "hierskm": (GRAY_L, (0, (4, 2))), "faisstwol": (GRAY_D, "-"),
           "hslskm": (ORANGE, "-"), "horfive": (CORAL, "-"), "wsthree": (OLIVE, "-"),
           "skmdef": (ROSE, "-"), "faissivf": (TEAL, "-"), "faissfull": (TEAL, (0, (4, 2))), "skmlloyd": (GREEN, "-")}
# 集合：固定颜色（PALETTE8 的已验证顺序）+ 固定标记
DIM = {"deep10m": 96, "sift": 128, "glove200": 200, "yahoo": 384, "gist": 960, "mxbai": 1024, "dbpedia": 1536, "wiki": 3072}
DSCOL = dict(zip(["sift", "glove200", "yahoo", "gist", "mxbai", "dbpedia", "wiki", "deep10m"], PALETTE8))
DSMARK = {"deep10m": "p", "sift": "o", "glove200": "s", "yahoo": "D", "gist": "^", "mxbai": "v", "dbpedia": "<", "wiki": ">"}


def ds_legend(handles, labels):
    """集合图例：与表格相同的集合顺序。"""
    return handles, labels
FAMILIES = [("CertRefine-H", ["oursHL", "oursHone", "oursHfive"]), ("CertRefine-B", ["oursBL", "oursBfive"]), ("HSL-SKM", ["hslskm"]),
            ("HOR", ["horone", "hortwo", "horfive"]), ("Lloyd-WS", ["wsone", "wstwo", "wsthree"]), ("SKM-default", ["skmdef"]),
            ("FAISS", ["faissivf", "faissfull"]), ("SKM-Lloyd", ["skmlloyd"]), ("Hier-SKM", ["hierskm"]), ("FAISS-2L", ["faisstwol"]), ("MiniBatch", ["minibatch"])]
DELTA = 0.005          # 等价检验的边界：±0.5 个召回点
N_BOOT = 2000


def tex(macro):
    return "\\" + macro


def famkey(fam):
    return fam.lower().replace("-", "")


def write_macros():
    lines = [f"% 由 experiments/analysis/make_assets.py 自动生成（RENAME_B={RENAME_B}），请勿手工修改",
             r"\newcommand{\oursH}{CertRefine-H\xspace}", r"\newcommand{\oursB}{CertRefine-B\xspace}",
             r"\newcommand{\faisssampled}{FAISS-sampled\xspace}"]
    for (tag, ep), (macro, printed, fam, grp) in NAMES.items():
        lines.append(r"\newcommand{\%s}{%s\xspace}" % (macro, printed))
    write_gen("macros.tex", "\n".join(lines) + "\n")


R = {ds: L.runs(ds) for ds in DS if os.path.isdir(os.path.join(L.RES, ds))}
for _ds in DS:
    if _ds not in R:
        missing(f"results directory {_ds}")


def entry(ds, m, e):
    return R.get(ds, {}).get((m, e))


def stat(ds, m, e, key="recall@scan0.01"):
    en = entry(ds, m, e)
    if not en:
        return None
    tm, ts, vm, vs, k = L.agg(en, key)
    return dict(t=tm, t_sd=ts, r=vm, r_sd=vs, k=k)


def sstat(ds, short, key="recall@scan0.01"):
    return stat(ds, *SHORT[short], key=key)


def dim_of(ds):
    en = entry(ds, "cr", 5)
    for run in (en or {}).values():
        return int(run["refine"]["d"])
    return None


def refine_json(ds, tag, seed):
    p = os.path.join(L.RES, ds, f"{tag}_s{seed}.refine.json")
    return json.load(open(p)) if os.path.exists(p) else None


def join_and(xs):
    """["a", "b", "c"] -> "a, b and c"；空 -> "none"。"""
    xs = list(xs)
    if not xs:
        return "none"
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def geo(a):
    a = [x for x in a if x and not np.isnan(x)]
    return float(np.exp(np.mean(np.log(a)))) if a else float("nan")


def caption_tex(text):
    """表题（数据库论文的写法）：一个简短的名词短语，通常 3–12 个词，句号结尾；方法、图例、种子数与解释都写在正文里，
    表内标记的图例至多一行，放在表下（table(..., note=...)）。"""
    words = len(re.sub(r"\$[^$]*\$", "x", text).split())
    if not (2 <= words <= 12) or not text.endswith("."):
        print("WARNING caption style:", words, "words:", text)
    return r"\caption{" + text + "}"


def note_tex(note):
    """表下的一行图例（\\footnotesize，紧贴 tabular，在浮动体内）。"""
    return r"\par\vspace{2pt}{\footnotesize " + note + r"\par}"


def table(env, caption, label, colspec, header, body, tabcolsep="4pt", adjust=True, note=None):
    """caption: 表题文字（不含 \\caption，见 caption_tex）；label: 如 "tab:main"；note: 可选的一行图例（只解释表内标记）。"""
    lines = [rf"\begin{{{env}}}[pos=!tb]", r"\centering\small", caption_tex(caption), rf"\label{{{label}}}", rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}"]
    # \fittable（导言区定义）：窄表用 tabular* 均匀撑满 \linewidth，宽表整体缩到 \linewidth
    lines += [rf"\fittable{{{colspec}}}{{", r"\toprule"] + header + [r"\midrule"] + body
    lines += [r"\bottomrule", r"}"]
    if note:
        lines.append(note_tex(note))
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


def grouped_rows(rows, ncol, titles=None):
    """rows: [(分组, 单元格列表)] -> 行文本；分组变化处插入小标题（titles: 分组 -> 标题，缺省为 GROUP_TITLE）。"""
    titles = titles or GROUP_TITLE
    out, cur = [], None
    for g, cells in rows:
        if g != cur:
            if cur is not None:
                out.append(r"\midrule")
            out.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{titles[g]}}}}} \\")
            cur = g
        out.append(" & ".join(cells) + r" \\")
    return out


# 表下图例（只解释标记；检验方法、种子数与定义写在正文里）
MARKS_H5 = r"$^{\blacktriangle}$/$^{\triangledown}$: significantly higher/lower than \oursHfive; $^{=}$: equivalent within $\pm$\V{stat.delta.pts} points"
DAGGER_HSL = r"$^{\dagger}$Batch Lloyd without certificates."


# ---------------------------------------------------------------- 1. 统计：两层自助法 + 每个数据集内对全部对照做 Holm
_CMP = {}


def compare(ds, a_key, b_key, b=L.B1):
    """两层自助法比较 a 与 b（per-seed × per-query 召回，a 减 b），不做多重校正；返回 boot_diff 的结果或 None。
    对称缓存：(b, a) 的结果由 (a, b) 取负得到，保证同一对方法在不同参照行里用的是同一个检验。"""
    key = (ds, a_key, b_key, b)
    if key in _CMP:
        return _CMP[key]
    rkey = (ds, b_key, a_key, b)
    if rkey in _CMP and _CMP[rkey] is not None:
        r = _CMP[rkey]
        _CMP[key] = dict(diff=-r["diff"], lo95=-r["hi95"], hi95=-r["lo95"], lo90=-r["hi90"], hi90=-r["lo90"], p=r["p"], equiv=r["equiv"])
        return _CMP[key]
    ea, eb = entry(ds, *a_key), entry(ds, *b_key)
    if not ea or not eb:
        _CMP[key] = None
        return None
    A, B = L.per_seed_pq(ea, b), L.per_seed_pq(eb, b)
    if A is None or B is None or A.shape[1] != B.shape[1]:
        _CMP[key] = None
        return None
    _CMP[key] = L.boot_diff(A, B, n_boot=N_BOOT, delta=DELTA)
    return _CMP[key]


def classify(bt, p_adj):
    """显著性优先：Holm 校正后 p<0.05 为 W/L；否则 90% 区间落在 ±DELTA 内为 E，再否则 I。"""
    if p_adj < 0.05:
        return "W" if bt["diff"] > 0 else "L"
    return "E" if bt["equiv"] else "I"


REFS = [("oursHfive", "cr5"), ("oursHL", "crl"), ("oursHone", "cr1")]   # (参照宏名, 键短名)；胜/等价/负表的三行
REFS_ALL = REFS + [("oursBL", "bl")]   # 统计与键：B-L 也作参照（决策表的"同样好的替代"需要它）


def stats_main():
    """每个参照 R ∈ {H-5, H-L, H-1} × 每个数据集：对 MAIN 里的其余全部方法做两层自助法，并在该集合内 Holm 校正。"""
    cmp, fam = {}, {}
    for ref, _ in REFS_ALL:
        rk = BY_MACRO[ref]
        cmp[ref] = {}
        for ds in DS:
            if not entry(ds, *rk):
                missing(f"{ds} {PRINTED[ref]} (reference for statistics)")
                continue
            res = []
            for tag, ep, m, g in MAIN:
                if m == ref:
                    continue
                bt = compare(ds, rk, (tag, ep))
                if bt is None:
                    if ref == "oursHfive":
                        missing(f"{ds} {PRINTED[m]} (main comparison)")
                    continue
                res.append((m, bt))
            if not res:
                continue
            adj = L.holm([bt["p"] for _, bt in res])
            cmp[ref][ds] = {}
            for (m, bt), pa in zip(res, adj):
                sig = bool(pa < 0.05)
                cmp[ref][ds][m] = dict(diff=100 * bt["diff"], ci90=[100 * bt["lo90"], 100 * bt["hi90"]], ci95=[100 * bt["lo95"], 100 * bt["hi95"]],
                                       p_holm=float(pa), sig=sig, equiv=bool(bt["equiv"]), cls=classify(bt, pa), family=len(res))
            if ref == "oursHfive":
                fam[ds] = len(res)
    SUMMARY["cmp"] = cmp
    for ds, n in fam.items():
        setv(f"fam.main.{ds}", n)
    if fam:
        setv("fam.main.min", min(fam.values()))
        setv("fam.main.range", str(min(fam.values())) if min(fam.values()) == max(fam.values())
             else f"{min(fam.values())}--{max(fam.values())}")
        setv("fam.main.max", max(fam.values()))
    esig = [(ref, ds, m, c["diff"]) for ref, _ in REFS for ds in cmp.get(ref, {}) for m, c in cmp[ref][ds].items() if c["sig"] and c["equiv"]]
    setv("wel.esig.count", len(esig))
    setv("wel.esig.cr5.count", sum(1 for e in esig if e[0] == "oursHfive"))
    setv("wel.esig.list", "; ".join(f"{PRINTED[ref]} vs {PRINTED[m]} on {L.DSNAME[ds]} ({d:.2f})" for ref, ds, m, d in esig) or "none")
    setv("stat.nboot", N_BOOT)
    setv("stat.delta.pts", 100 * DELTA, 1)
    ks = [stat(ds, tag, ep)["k"] for ds in DS for tag, ep, m, g in MAIN if stat(ds, tag, ep)]
    if ks:
        setv("stat.seeds.main", min(ks))
        setv("stat.seeds.main.max", max(ks))
    ks = [len(en) for ds in DS for (m, e), en in R.get(ds, {}).items() if (m.startswith("cr_K") or m == "cr_snap" or m == "single") and e == 5]
    if ks:
        setv("stat.seeds.sens", min(ks))
    return cmp


# ---------------------------------------------------------------- 主表：recall@1% 与构建时间（含显著性与等价标记）
def marks_of(c):
    """c = SUMMARY['cmp'][ref][ds][m]（参照减对照）：对照显著更高 ▲、更低 ▽；90% 区间在 ±0.5 内加 =。"""
    if not c:
        return ""
    sup = ""
    if c["sig"]:
        sup += "\\blacktriangle" if c["diff"] < 0 else "\\triangledown"
    if c["equiv"]:
        sup += "="
    return r"$^{" + sup + "}$" if sup else ""


def tstr(t):
    return fmt(t, 1) if t < 100 else f"{t:.0f}"


def sdstr(sd):
    """标准差：两位有效数字，最多两位小数（0.06、0.64、2.6、33、140）。"""
    if sd >= 10:
        return f"{sd:.0f}"
    if sd >= 1:
        return f"{sd:.1f}"
    return f"{sd:.2f}"


def main_tables(cmp):
    S = {ds: {m: stat(ds, tag, ep) for tag, ep, m, g in MAIN + [HSLROW]} for ds in DS}
    best = {}
    for ds in DS:
        vals = [(S[ds][m] or {}).get("r", np.nan) for _, _, m, _ in MAIN]
        best[ds] = np.nanmax(vals) if not np.all(np.isnan(vals)) else np.nan
    c5 = cmp.get("oursHfive", {})
    rows_r, rows_t, rows_sd = [], [], []
    rel_sd = []   # (SD/均值 %, 方法, 集合)，正文概括时间的离散程度
    for tag, ep, m, g in MAIN + [HSLROW]:
        lab = tex(m) + (r"\unskip$^{\dagger}$" if m == "hslnocert" else "")   # \unskip: 宏的 \xspace 在 $ 前会插入空格
        cr, ct, csd = [lab], [lab], [lab]
        for ds in DS:
            s = S[ds][m]
            if not s or np.isnan(s["r"]):
                cr.append("--")
                ct.append("--")
                csd.append("--")
                missing(f"{ds} {PRINTED[m]} (main table)")
                continue
            txt = f"{100 * s['r']:.2f}"
            if abs(s["r"] - best[ds]) < 5e-7 and m != "hslnocert":
                txt = r"\textbf{" + txt + "}"
            cr.append(txt + marks_of(c5.get(ds, {}).get(m)))
            t_ = tstr(s["t"])
            if not np.isnan(s["t_sd"]):
                t_ += r"{\scriptsize$\pm$" + sdstr(s["t_sd"]) + "}"
                if m in MAIN_MACROS:
                    rel_sd.append((100 * s["t_sd"] / s["t"], m, ds))
            ct.append(t_)
            csd.append("--" if np.isnan(s["t_sd"]) else fmt(s["t_sd"], 2 if s["t_sd"] < 10 else 1))
            if not np.isnan(s["t_sd"]):
                setv(f"sd.{ds}.{m}.t", s["t_sd"], 2 if s["t_sd"] < 10 else 1)
        if m != "hslnocert":
            rows_r.append((g, cr))
        rows_t.append((g, ct))
        rows_sd.append((g, csd))
    # HSL (no cert) 只进时间表：放在"同一分区的其它精修"组的开头
    ins = next(i for i, (g, c) in enumerate(rows_t) if c[0].startswith(tex("hslnocert")))
    hrow_t, hrow_sd = rows_t.pop(ins), rows_sd.pop(ins)
    pos = next(i for i, (g, c) in enumerate(rows_t) if g == "refine")
    rows_t.insert(pos, hrow_t)
    rows_sd.insert(pos, hrow_sd)
    hdr = ["Method & " + " & ".join(L.DSNAME[d] for d in DS) + r" \\"]
    ncol = 1 + len(DS)
    write_gen("tab_main_recall.tex", table(
        "table*", r"Recall@10 (\%) at a 1\% scan budget.", "tab:main", "l" + "r" * len(DS), hdr, grouped_rows(rows_r, ncol),
        note=MARKS_H5 + "; bold: best."))
    write_gen("tab_main_time.tex", table(
        "table*", "End-to-end build time (s).", "tab:time", "l" + "r" * len(DS), hdr, grouped_rows(rows_t, ncol),
        tabcolsep="3pt", note=DAGGER_HSL + r" Mean{\scriptsize$\pm$SD} over seeds."))
    if rel_sd:
        cert = [x for x in rel_sd if x[1] in ("oursHL", "oursHone", "oursHfive", "oursBL", "oursBfive")]
        setv("sd.rel.cert.max", max(x[0] for x in cert), 1)
        top = max(rel_sd)
        setv("sd.rel.max", top[0], 1)
        setv("sd.rel.max.method", tex(top[1]))
        setv("sd.rel.max.ds", L.DSNAME[top[2]])
        big = sorted({x[1] for x in rel_sd if x[0] > 5})
        setv("sd.rel.gt5.list", ", ".join(tex(m) for m in big) or "none")
    sds = [S[ds]["oursHfive"]["r_sd"] for ds in DS if S[ds]["oursHfive"] and not np.isnan(S[ds]["oursHfive"]["r_sd"])]
    if sds:
        setv("tab.cr5.maxsd", 100 * max(sds), 2)
    return S


# ---------------------------------------------------------------- 正文数字：RQ1（速度、胜/等价/负表、逐数据集数值、成对范围）
SP_REFS = ["cr5", "cr1", "crl", "bl"]
SP_OTHERS = ["fivf", "ffull", "lloyd", "hor", "hor1", "hor2", "hslskm", "ws1", "ws2", "ws3", "skmdef", "hier", "bl", "b5", "tlc"]
WEL_COLS = ["hierskm", "faisstwol", "oursBL", "oursBfive", "hslskm", "horone", "horfive", "wsthree", "skmdef", "faissivf", "faissfull", "skmlloyd"]   # 只用于键（旧别名）
T6_REFS = [("oursBL", "bl"), ("oursHL", "crl"), ("oursHone", "cr1"), ("oursHfive", "cr5")]   # Table 6 的列组：本文工作点，按构建成本
T6_ROWS = ["hierskm", "faisstwol", "hslskm", "horone", "hortwo", "horfive", "wsone", "wstwo", "wsthree", "skmdef", "faissivf", "faissfull", "skmlloyd"]
OLD_WEL = {"oursBL": "hslpcertl", "horfive": "hor", "wsthree": "lloydws3", "skmdef": "skmdefault"}   # 旧键名别名
PAIRS = [("tlc", "cr5"), ("tlc", "crl"), ("tlc", "bl"), ("hor1", "bl"), ("hor2", "crl"), ("hor2", "cr1"), ("hor1", "crl"), ("ws1", "cr1"), ("ws2", "cr5"), ("ws3", "cr5"), ("cr1", "hsl"), ("cr1", "b5"),
         ("crl", "bl"), ("b5", "cr5"), ("hslskm", "cr5"), ("hor", "cr5")]


def speed_keys(prefix, a, b, nd=1):
    """prefix.min/max/geo/faster = t(b)/t(a) 逐数据集（未取整的种子均值），两个方向都由此函数算，绝不由取整后的键取倒数。"""
    r_ = []
    for ds in DS:
        sa, sb = sstat(ds, a), sstat(ds, b)
        if sa and sb and isnum(sa["t"]) and isnum(sb["t"]):
            r_.append(sb["t"] / sa["t"])
    if r_:
        setv(f"{prefix}.min", min(r_), nd)
        setv(f"{prefix}.max", max(r_), nd)
        setv(f"{prefix}.geo", geo(r_), nd)
        setv(f"{prefix}.faster", sum(1 for x in r_ if x > 1))
    return r_


def pair_range(prefix, a, b):
    """成对范围：prefix.r.min/max（召回差 a−b，点）、.t.min/max（时间比 a/b）、.tie/.above/.below.list（|diff|≤0.01 点为平）。"""
    rs, ts, tie, ab, be, tf, tsl = [], [], [], [], [], [], []
    for ds in DS:
        sa, sb = sstat(ds, a), sstat(ds, b)
        if not sa or not sb or not isnum(sa["r"]) or not isnum(sb["r"]):
            continue
        d = 100 * (sa["r"] - sb["r"])
        rs.append(d)
        ts.append(sa["t"] / sb["t"])
        (tie if abs(d) <= 0.01 else (ab if d > 0 else be)).append(L.DSNAME[ds])
        (tf if sa["t"] < sb["t"] else tsl).append(L.DSNAME[ds])
    if rs:
        setv(f"{prefix}.r.min", min(rs), 2)
        setv(f"{prefix}.r.max", max(rs), 2)
        setv(f"{prefix}.t.min", min(ts), 2)
        setv(f"{prefix}.t.max", max(ts), 2)
        setv(f"{prefix}.r.pos", sum(1 for x in rs if x > 0))
        setv(f"{prefix}.tie.list", ", ".join(tie) or "none")
        setv(f"{prefix}.above.list", ", ".join(ab) or "none")
        setv(f"{prefix}.below.list", ", ".join(be) or "none")
        setv(f"{prefix}.tfaster.list", ", ".join(tf) or "none")   # a 比 b 快的集合
        setv(f"{prefix}.tslower.list", ", ".join(tsl) or "none")
        setv(f"{prefix}.tfaster.count", len(tf))
        setv(f"{prefix}.n", len(rs))


def rq1_numbers(cmp, S):
    per = {ds: {k: sstat(ds, k) for k in SHORT} for ds in DS if sstat(ds, "cr5")}
    if not per:
        missing("CertRefine-H-5 on every collection (RQ1 numbers)")
        return
    # --- 一轮精修相对分层版：增益与显著性（来自主统计：H-1 参照，对 Hier-SKM）
    g1 = [(ds, c["hierskm"]) for ds, c in cmp.get("oursHone", {}).items() if "hierskm" in c]
    if g1:
        setv("rq1.gain1.min", min(c["diff"] for _, c in g1), 1)
        setv("rq1.gain1.max", max(c["diff"] for _, c in g1), 1)
        setv("rq1.gain1.sig.count", sum(1 for _, c in g1 if c["cls"] == "W"))
        setv("rq1.gain1.n", len(g1))
    hg = [100 * (max(x["r"] for x in (p["fivf"], p["ffull"], p["lloyd"]) if x) - p["hier"]["r"]) for p in per.values()
          if p["hier"] and any(p[k] for k in ("fivf", "ffull", "lloyd"))]
    if hg:
        setv("rq1.hiergap.min", min(hg), 1)
        setv("rq1.hiergap.max", max(hg), 1)
        if "tab.cr5.maxsd" in V:
            setv("stat.delta.mingap", min(hg), 1)
            setv("stat.delta.maxsd.ratio", 0.5 / float(V["tab.cr5.maxsd"]), 1)
    # --- 速度（端到端）：两个方向都从未取整的种子均值算
    for ref in SP_REFS:
        for oth in SP_OTHERS:
            if oth == ref:
                continue
            speed_keys(f"sp.{ref}.{oth}", ref, oth)
            speed_keys(f"sp.{oth}.{ref}", oth, ref, nd=2)
    # --- H-5 相对同一分区的其它精修（1M 级的七个数据集）：t(H-5)/t(rival)
    cv = [per[ds]["cr5"]["t"] / per[ds][rv]["t"] for ds in per if ds != "deep10m" for rv in ("hslskm", "hor", "b5", "ws3") if per[ds][rv]]
    if cv:
        setv("rq1.cr5.cost.vs.refine.min", min(cv), 2)
        setv("rq1.cr5.cost.vs.refine.max", max(cv), 2)
    # --- 胜/等价/负表（全部来自主统计 cmp）
    wel = {}
    for ref, rs in REFS_ALL:
        for ds, cs in cmp.get(ref, {}).items():
            for m, c in cs.items():
                wel.setdefault((ref, m), {})[ds] = dict(cls=c["cls"], diff=c["diff"], ci90=c["ci90"], p_holm=c["p_holm"])
    SUMMARY["wel"] = {f"{k[0]}|{k[1]}": v for k, v in wel.items()}
    rs_of = dict(REFS_ALL)
    for (ref, m), cells in wel.items():
        nm = f"wel.{rs_of[ref]}.{m.lower()}"
        cl = [v["cls"] for v in cells.values()]
        for k in "WELI":
            setv(f"{nm}.{k}", cl.count(k))
        for k in "WELI":
            setv(f"{nm}.{k}.list", ", ".join(L.DSNAME[ds] for ds, v in cells.items() if v["cls"] == k) or "none")
            dd = [abs(v["diff"]) for v in cells.values() if v["cls"] == k]
            if dd:
                setv(f"{nm}.{k}.min", min(dd), 2)
                setv(f"{nm}.{k}.max", max(dd), 2)
        for ds, v in cells.items():
            setv(f"{nm}.{ds}.cls", v["cls"])
            setv(f"{nm}.{ds}.diff", v["diff"], 2)
    # 表：行为其余方法（按主表分组），列组为 H-5 / H-L / H-1：端到端加速比（数据集上的几何平均）、H 点更快的数据集数、W/E/L/I
    welrow = {m: [tex(m)] for m in WEL_COLS}
    for ref, rs in REFS:
        for m in WEL_COLS:
            if m == ref:
                welrow[m] += ["--"] * 3
                continue
            r_ = [S[ds][m]["t"] / S[ds][ref]["t"] for ds in DS if S[ds].get(m) and S[ds].get(ref) and isnum(S[ds][m]["t"]) and isnum(S[ds][ref]["t"])]
            welrow[m] += [f"{geo(r_):.1f}$\\times$", str(sum(1 for x in r_ if x > 1))] if r_ else ["--", "--"]
            cells = wel.get((ref, m), {})
            cl = [v["cls"] for v in cells.values()]
            welrow[m].append("/".join(str(cl.count(k)) for k in "WELI") if cl else "--")
            names = [f"wel.{rs}.{m.lower()}"] + ([f"wel.{rs}.{OLD_WEL[m]}"] if m in OLD_WEL else [])
            for nm in names:
                for k in "WELI":
                    setv(f"{nm}.{k}", cl.count(k))
                for k in "WL":
                    dsl = [L.DSNAME[ds] for ds, v in cells.items() if v["cls"] == k]
                    setv(f"{nm}.{k}.list", ", ".join(dsl) or "none")
                    dd = [abs(v["diff"]) for v in cells.values() if v["cls"] == k]
                    if dd:
                        setv(f"{nm}.{k}.min", min(dd), 2)
                        setv(f"{nm}.{k}.max", max(dd), 2)
    # 表（正文 Table tab:wel）：列组为本文的四个工作点 B-L / H-L / H-1 / H-5（按构建成本），行为其余全部方法（按主表分组，
    # 不含本文自己的工作点）；每格：端到端加速比（数据集上的几何平均，括号内为本文工作点更快的数据集数）与 W/E/L/I
    rows6 = []
    for m in T6_ROWS:
        cells = [tex(m)]
        for ref, _ in T6_REFS:
            r_ = [S[ds][m]["t"] / S[ds][ref]["t"] for ds in DS if S[ds].get(m) and S[ds].get(ref) and isnum(S[ds][m]["t"]) and isnum(S[ds][ref]["t"])]
            cells.append(f"{geo(r_):.1f}$\\times$\\,({sum(1 for x in r_ if x > 1)})" if r_ else "--")
            cl = [v["cls"] for v in wel.get((ref, m), {}).values()]
            cells.append("/".join(str(cl.count(k)) for k in "WELI") if cl else "--")
        rows6.append((GROUP_OF[m], cells))
    write_gen("tab_wel.tex", table(
        "table*", "Summary of pairwise comparisons.", "tab:wel", "l" + "rc" * len(T6_REFS),
        [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{tex(ref)}}}" for ref, _ in T6_REFS) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(T6_REFS))),
         "Method & " + " & ".join(["speedup & W/E/L/I"] * len(T6_REFS)) + r" \\"],
        grouped_rows(rows6, 1 + 2 * len(T6_REFS)), tabcolsep="4pt",
        note=r"Speedup: build time of the method divided by that of the \ours point, geometric mean over the collections"
             r" (in parentheses: collections on which the \ours point is faster). W/L: recall of the \ours point significantly"
             r" higher/lower; E: equivalent within $\pm$\V{stat.delta.pts} points; I: inconclusive."))

    # --- 最便宜的认证构建 B-L 相对其它方法（摘要、引言、6.1、结论的措辞都由这些键渲染）
    def tr(ds, m, ref="oursBL"):
        a, b = S[ds].get(m), S[ds].get(ref)
        if not a or not b or not isnum(a["t"]) or not isnum(b["t"]):
            return None
        return a["t"] / b["t"]          # > 1：B-L 更快

    def lset(ref, ms, k="L"):
        return join_and(L.DSNAME[ds] for ds in DS if any(wel.get((ref, m), {}).get(ds, {}).get("cls") == k for m in ms))
    ref_group = [m for m in T6_ROWS if GROUP_OF[m] in ("hier", "refine") and m != "hierskm"]
    flat_group = [m for m in T6_ROWS if GROUP_OF[m] in ("sampled", "full")]
    fall = [m for m in T6_ROWS if m != "hierskm" and all((tr(ds, m) or 0) > 1 for ds in DS)]
    fr = [m for m in fall if m in ref_group]
    setv("bl.fasterall.ref.list", join_and(tex(m) for m in fr))
    setv("bl.fasterall.ref.L.list", lset("oursBL", fr))
    setv("bl.fasterall.flat.list", join_and(tex(m) for m in fall if m in flat_group))
    setv("bl.hsltlc.L.list", lset("oursBL", ["hslskm", "faisstwol"]))
    for m in ("hslskm", "faisstwol"):
        if m not in fall:
            print("ERROR claim (abstract): B-L is not faster than", m, "on every collection")
    riv = [(m, sum(1 for ds in DS if (tr(ds, m) or 99) < 1)) for m in T6_ROWS if m != "hierskm"]
    riv = [(m, k) for m, k in riv if k >= 2]
    setv("bl.rivals.list", join_and(f"{tex(m)} ({k} of {len(DS)})" for m, k in riv))
    setv("bl.rivals.deep.slower", "yes" if all((tr("deep10m", m) or 0) > 1 for m, _ in riv) else "no")
    if V["bl.rivals.deep.slower"] != "yes":
        print("ERROR claim (6.1): a method faster than B-L on one-million collections is also faster on Deep-10M")
    setv("bl.hor12.L.list", lset("oursBL", ["horone", "hortwo"]))
    deep_ref = [m for m in ref_group if m != "faisstwol"]
    dr = [tr("deep10m", m) for m in deep_ref if tr("deep10m", m)]
    if dr:
        setv("deep.bl.vsrefine.min", min(dr), 1)
        setv("deep.bl.vsrefine.max", max(dr), 1)
        cls = [wel.get(("oursBL", m), {}).get("deep10m", {}).get("cls") for m in deep_ref]
        ok = all(c in ("E", "W") for c in cls) and min(dr) > 1
        setv("deep.bl.vsrefine.ok", "yes" if ok else "no")
        if not ok:
            print("ERROR claim (abstract): at ten million vectors B-L is not faster at equivalent recall than every refinement", cls, dr)
    # 6.1 的总起句是硬写的定性断言：数据一旦不再支持就报错（不渲染数字，只校验）
    certs = ["oursHL", "oursHone", "oursHfive", "oursBL", "oursBfive"]
    for m in ("faissivf", "faissfull", "skmlloyd"):
        for c_ in certs:
            if not all((tr(ds, m, c_) or 0) > 1 for ds in DS):
                print("ERROR claim (6.1): certified point", c_, "is not faster than", m, "on every collection")
    for ds in DS:
        a_, b_ = S[ds].get("faisstwol"), S[ds].get("oursHfive")
        if a_ and b_ and isnum(a_["r"]) and isnum(b_["r"]) and a_["r"] >= b_["r"]:
            print("ERROR claim (6.1): FAISS-2L is not less accurate than CertRefine-H-5 on", ds)
        if (tr(ds, "faisstwol") or 0) <= 1:
            print("ERROR claim (6.1): FAISS-2L is not slower than CertRefine-B-L on", ds)
    if any(v["cls"] == "L" for v in wel.get(("oursHfive", "skmdef"), {}).values()):
        print("ERROR claim (6.1): SKM-default is significantly more accurate than CertRefine-H-5 somewhere")
    if sum(1 for ds in DS if (tr(ds, "skmdef", "oursHfive") or 9) < 1) < len(DS) // 2 + 1:
        print("ERROR claim (6.1): SKM-default is not usually faster than CertRefine-H-5")
    ltl = [L.DSNAME[ds] for ds, v in wel.get(("oursHfive", "faisstwol"), {}).items() if v["cls"] == "L"]
    setv("claim.cr5.tlc.less", "never significantly less accurate" if not ltl else "significantly less accurate on " + join_and(ltl))

    # --- H-5 与全量方法：半个点以内 / 更高 / 最大差距；对更好的那个全量方法的 W/E/L/I 类别
    within, above, gaps, cls_ = [], [], {}, {}
    for ds, p in per.items():
        full = [(k, p[k]["r"]) for k in ("ffull", "lloyd") if p[k] and isnum(p[k]["r"])]
        if not full:
            continue
        bk, br = max(full, key=lambda kv: kv[1])
        gaps[ds] = 100 * (br - p["cr5"]["r"])
        if p["cr5"]["r"] >= br - DELTA:
            within.append(L.DSNAME[ds])
        if p["cr5"]["r"] > br:
            above.append(L.DSNAME[ds])
        bm = MACRO_OF_SHORT[bk]
        c = cmp.get("oursHfive", {}).get(ds, {}).get(bm)
        setv(f"rq1.cr5.vsbestfull.{ds}.name", tex(bm))
        if c:
            cls_[ds] = c["cls"]
            setv(f"rq1.cr5.vsbestfull.{ds}.cls", c["cls"])
    if gaps:
        setv("rq1.cr5.within.count", len(within))
        setv("rq1.cr5.within.n", len(gaps))
        setv("rq1.cr5.within.list", ", ".join(within) or "none")
        setv("rq1.cr5.above.list", ", ".join(above) or "none")
        setv("rq1.cr5.above.count", len(above))
        worst = max(gaps, key=gaps.get)
        setv("rq1.cr5.gap.max", gaps[worst], 2)
        setv("rq1.cr5.gap.max.ds", L.DSNAME[worst])
        for k in "WELI":
            l_ = [L.DSNAME[ds] for ds, c in cls_.items() if c == k]
            setv(f"rq1.cr5.vsbestfull.{k}.count", len(l_))
            setv(f"rq1.cr5.vsbestfull.{k}.list", ", ".join(l_) or "none")
    # --- Pareto 前沿（公平：每个方法族都有自己的工作点）
    front = {}
    for ds in per:
        pts = {}
        for fam, ms in FAMILIES:
            for m in ms:
                s_ = S[ds].get(m) or stat(ds, *BY_MACRO[m])
                if s_ and isnum(s_["t"]) and isnum(s_["r"]):
                    pts[(fam, m)] = (s_["t"], s_["r"])
        front[ds] = [a for a in pts if not any(pts[b][0] <= pts[a][0] and pts[b][1] >= pts[a][1] and
                                               (pts[b][0] < pts[a][0] or pts[b][1] > pts[a][1]) for b in pts if b != a)]
    SUMMARY["pareto_front"] = {ds: [f"{a[0]}:{PRINTED[a[1]]}" for a in v] for ds, v in front.items()}
    for fam, ms in FAMILIES:
        setv(f"pareto.{famkey(fam)}.count", sum(1 for v in front.values() if any(a[0] == fam for a in v)))
    merged = sum(1 for v in front.values() if any(a[0] in ("CertRefine-H", "CertRefine-B") for a in v))
    setv("pareto.certrefine.merged.count", merged)
    setv("pareto.certrefine.count", merged)
    setv("pareto.n", len(front))
    if any(a[0] == "MiniBatch" for v in front.values() for a in v):
        setv("mb.pareto.count", sum(1 for v in front.values() if any(a[0] == "MiniBatch" for a in v)))
        for ds, v in front.items():
            if any(a[0] == "MiniBatch" for a in v):
                print("WARNING minibatch on the Pareto front of", ds)
    # --- 成对范围（正文措辞用）
    for a, b in PAIRS:
        pair_range(f"{a}.vs.{b}", a, b)
    for k in ("r.min", "r.max", "t.min", "t.max", "r.pos"):
        if f"crl.vs.bl.{k}" in V:
            V[f"crl.vs.hslcl.{k}"] = V[f"crl.vs.bl.{k}"]
    # --- HOR-5 / HSL-SKM 比 H-5 快的数据集：个数、最小维度、最大比值
    for rv, key in (("hor", "hor"), ("hslskm", "hslskm")):
        f_ = [(ds, per[ds]["cr5"]["t"] / per[ds][rv]["t"]) for ds in per if per[ds][rv] and per[ds][rv]["t"] < per[ds]["cr5"]["t"]]
        setv(f"rq1.{key}.faster.count", len(f_))
        if f_:
            setv(f"rq1.{key}.faster.max", max(x for _, x in f_), 2)
            dims = [dim_of(ds) for ds, _ in f_ if dim_of(ds)]
            if dims:
                setv(f"rq1.{key}.faster.dim", min(dims))
            setv(f"rq1.{key}.faster.list", ", ".join(L.DSNAME[ds] for ds, _ in f_))
    # --- Deep-10M：接近饱和的召回范围与五个 CertRefine 工作点的加速范围
    if "deep10m" in per:
        rr = [S["deep10m"][m]["r"] for _, _, m, _ in MAIN if S["deep10m"].get(m) and isnum(S["deep10m"][m]["r"])]
        if rr:
            setv("rq1.deep.r.min", 100 * min(rr), 2)
            setv("rq1.deep.r.max", 100 * max(rr), 2)
        ours = [per["deep10m"][k]["t"] for k in ("crl", "cr1", "cr5", "bl", "b5") if per["deep10m"][k]]
        for rv in ("ffull", "fivf", "lloyd"):
            if per["deep10m"][rv] and ours:
                setv(f"rq1.deep.speed.{rv}.min", per["deep10m"][rv]["t"] / max(ours), 1)
                setv(f"rq1.deep.speed.{rv}.max", per["deep10m"][rv]["t"] / min(ours), 1)
    # --- 证书对批量 Lloyd（HSL）的加速（分区逐点相同），与 B-5 相对 H-5 的节省
    hs = [per[ds]["hsl"]["t"] / per[ds]["b5"]["t"] for ds in per if per[ds]["hsl"] and per[ds]["b5"]]
    if hs:
        setv("rq3.hslcert.speed.min", min(hs), 2)
        setv("rq3.hslcert.speed.max", max(hs), 2)
    sv = [100 * (1 - per[ds]["b5"]["t"] / per[ds]["cr5"]["t"]) for ds in per if per[ds]["b5"]]
    if sv:
        setv("rq3.hslc.saving.min", min(sv), 0)
        setv("rq3.hslc.saving.max", max(sv), 0)
    # --- 平衡性（cv，种子均值）：H-5 对 FAISS-IVF / FAISS-full / SKM-Lloyd
    cnt_bal, n_bal = 0, 0
    for ds in per:
        c5 = stat(ds, "cr", 5, "cv")
        fl = [stat(ds, m, None, "cv") for m in ("faiss_ivf", "faiss", "lloyd")]
        fl = [x for x in fl if x and isnum(x["r"])]
        if c5 and fl:
            n_bal += 1
            cnt_bal += int(c5["r"] < min(x["r"] for x in fl))
    setv("bal.count", cnt_bal)
    setv("bal.n", n_bal)
    cvh = [stat(ds, m, e, "cv")["r"] for ds in per for m, e in (("hier", None), ("cr", 5), ("cr_labels", 5), ("hsl", 5), ("hor", None))
           if stat(ds, m, e, "cv")]
    cvf = [stat(ds, m, None, "cv")["r"] for ds in per for m in ("faiss_ivf", "faiss", "lloyd", "skm_default") if stat(ds, m, None, "cv")]
    if cvh and cvf:
        setv("bal.hier.min", min(cvh), 2)
        setv("bal.hier.max", max(cvh), 2)
        setv("bal.flat.min", min(cvf), 2)
        setv("bal.flat.max", max(cvf), 2)
    # --- 逐数据集的数值（正文举例用）：宏名键 + 旧短名键
    for ds in DS:
        for tag, ep, m, g in MAIN + [HSLROW]:
            s = S[ds].get(m)
            if s and isnum(s["r"]):
                setv(f"ds.{ds}.{m}", 100 * s["r"], 2)
                setv(f"ds.{ds}.{m}.t", s["t"], 1 if s["t"] < 100 else 0)
        for k, m in MACRO_OF_SHORT.items():
            if f"ds.{ds}.{m}" in V:
                V[f"ds.{ds}.{k}"] = V[f"ds.{ds}.{m}"]
                V[f"ds.{ds}.{k}.t"] = V[f"ds.{ds}.{m}.t"]
    SUMMARY["rq1_per"] = {ds: {k: (v["t"], v["r"]) for k, v in p.items() if v} for ds, p in per.items()}


# ---------------------------------------------------------------- 图：Pareto（构建时间 vs recall），8 个小图
def pareto_front(pts):
    """pts: {lab: (time, recall)} → 不被支配的点，按时间升序。"""
    fr = [a for a in pts if not any(pts[b][0] <= pts[a][0] and pts[b][1] >= pts[a][1] and
                                    (pts[b][0] < pts[a][0] or pts[b][1] > pts[a][1]) for b in pts if b != a)]
    return sorted(fr, key=lambda a: pts[a][0])


def fig_pareto():
    """逐数据集：端到端构建时间 vs recall@10（1% 预算）。每个方法族的多个工作点连成一条线（每族都有自己的工作点，比较对等；
    所有族同样的标记大小与线宽）；灰色虚线是 Pareto 前沿；误差线为种子间标准差（时间与召回）。"""
    fig, axes = plt.subplots(2, 4, figsize=FS(4.2))
    handles = {}
    for ax, ds in zip(axes.ravel(), DS):
        style(ax)
        pts = {}
        for fam, ms in FAMILIES:
            col, ls = FAM_STYLE[fam]
            xs, ys = [], []
            for m in ms:
                s_ = stat(ds, *BY_MACRO[m])
                if not s_ or not isnum(s_["r"]) or not isnum(s_["t"]):
                    continue
                hollow = m in HOLLOW
                h = ax.errorbar(s_["t"], 100 * s_["r"], xerr=None if np.isnan(s_["t_sd"]) else s_["t_sd"],
                                yerr=None if np.isnan(s_["r_sd"]) else 100 * s_["r_sd"], fmt=MARKER.get(m, "o"), color=col,
                                ms=5.5 * {"*": 1.5, "P": 1.1, "p": 1.1}.get(MARKER.get(m, "o"), 1.0),
                                mfc="white" if hollow else col, mec=col if hollow else "white", mew=0.9 if hollow else 0.6,
                                elinewidth=0.6, capsize=0, zorder=3)
                handles.setdefault(fam, h)
                xs.append(s_["t"])
                ys.append(100 * s_["r"])
                pts[(fam, m)] = (s_["t"], 100 * s_["r"])
            if len(xs) > 1:
                ax.plot(xs, ys, ls=ls, color=col, lw=1.0, alpha=0.9, zorder=2)
        fr = pareto_front(pts)
        if len(fr) > 1:
            h = ax.step([pts[a][0] for a in fr], [pts[a][1] for a in fr], where="post", color=INK2, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
            handles.setdefault("Pareto front", h[0])
        ax.set_xscale("log")
        ax.set_title(L.DSNAME[ds], loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 3.0)))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    for ax in axes[:, 0]:
        ax.set_ylabel("recall@10 at 1% scan (%)")
    for ax in axes[-1]:
        ax.set_xlabel("build time (s)")
    order = [f for f, _ in FAMILIES if f in handles] + (["Pareto front"] if "Pareto front" in handles else [])
    top_legend(fig, [handles[k] for k in order], order, 5)
    fig.savefig(os.path.join(FIG, "fig_pareto.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- 图：recall 随扫描预算（五个种子的均值线 + 浅色 ±SD 带）
def fig_scan():
    """recall@10 与 H-5 之差随扫描预算的变化（各种子的曲线先在预算网格上线性插值，与正文 1% 处的插值方式相同；
    实线为种子均值，浅色带为 ±1 SD）。差值图能看清方法间零点几个点的差距，原始曲线图上它们会重叠。"""
    show = [(m,) + M_STYLE[m] for m in ("hslskm", "horfive", "wsthree", "skmdef", "faissivf", "faissfull", "skmlloyd", "hierskm")]
    grid = np.geomspace(0.0025, 0.04, 33)

    def curves(en):
        ys = []
        for s in L.SEEDS:
            if s in en and en[s]["scan"]:
                sc = en[s]["scan"]
                fr, rc = np.asarray(sc["scan_frac"]), np.asarray(sc["recall"])
                ys.append(np.interp(grid, fr, rc, left=np.nan, right=np.nan))
        if not ys:
            return None, None
        ys = 100 * np.asarray(ys)
        return ys.mean(axis=0), (ys.std(axis=0, ddof=1) if len(ys) > 1 else np.zeros_like(grid))

    fig, axes = plt.subplots(2, 4, figsize=FS(3.8))
    hier_q25 = []
    for ax, ds in zip(axes.ravel(), DS):
        style(ax)
        ref = entry(ds, "cr", 5)
        yr, _ = curves(ref) if ref else (None, None)
        if yr is None:
            ax.set_visible(False)
            continue
        ax.axhline(0, color=BLUE, lw=1.4, label=PRINTED["oursHfive"] + " (reference)")
        for m, col, ls in show:
            en = entry(ds, *BY_MACRO[m])
            y, sd = curves(en) if en else (None, None)
            if y is not None:
                ax.fill_between(100 * grid, y - yr - sd, y - yr + sd, color=col, alpha=0.07, lw=0)
                ax.plot(100 * grid, y - yr, ls=ls, color=col, lw=1.2, label=PRINTED[m])
                if m == "hierskm" and not np.isnan(y[0] - yr[0]):
                    hier_q25.append(yr[0] - y[0])
        ax.axvline(1, color=AXIS, lw=0.7, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_xlim(0.25, 4)
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(L.DSNAME[ds], loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    for ax in axes[:, 0]:
        ax.set_ylabel("recall difference (points)")
    for ax in axes[-1]:
        ax.set_xlabel("scan budget (%)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, h, lab, 5)
    fig.savefig(os.path.join(FIG, "fig_scan.pdf"))
    plt.close(fig)
    if hier_q25:
        setv("scan.hier.q25.max", max(hier_q25), 1)


# ---------------------------------------------------------------- RQ3：证书
def rq3_cert():
    skip = {}
    for ds in DS:
        en = [entry(ds, "cr", e) for e in range(1, 6)]
        if not all(en) or not all(42 in x for x in en):
            missing(f"{ds} cr_s42 five passes (skip rates)")
            continue
        skip[ds] = [en[e - 1][42]["hist"]["skipped_pct"] for e in range(1, 6)]
    if skip:
        setv("rq3.skip.p2.min", min(v[1] for v in skip.values()), 0)
        setv("rq3.skip.p2.max", max(v[1] for v in skip.values()), 0)
        setv("rq3.skip.p5.min", min(v[4] for v in skip.values()), 0)
        setv("rq3.skip.p5.max", max(v[4] for v in skip.values()), 0)
        lo = min(skip, key=lambda d: skip[d][1])
        setv("rq3.skip.lowest.ds", L.DSNAME[lo])
    SUMMARY["skip"] = skip
    # 标签起步（cr_labels_s42）的跳过率
    skl = {}
    for ds in DS:
        r = refine_json(ds, "cr_labels", 42)
        if r and len(r["hist"]) > 5:
            skl[ds] = [h["skipped_pct"] for h in r["hist"][1:6]]
    if skl:
        setv("rq3.skip.lab.p2.min", min(v[1] for v in skl.values()), 0)
        setv("rq3.skip.lab.p2.max", max(v[1] for v in skl.values()), 0)
        setv("rq3.skip.lab.p5.min", min(v[4] for v in skl.values()), 0)
        setv("rq3.skip.lab.p5.max", max(v[4] for v in skl.values()), 0)
    SUMMARY["skip_labels"] = skl
    # 与无证书运行逐轮 md5 一致性（全部有 cr_nocert 的种子）+ 实测时间比（seed 42）
    ident, speed_ref, speed_tot, single, speed_later = {}, {}, {}, {}, {}
    for ds in DS:
        lack = []
        for s in L.SEEDS:
            a, b = refine_json(ds, "cr", s), refine_json(ds, "cr_nocert", s)
            if a and b:
                ident[f"{ds}|{s}"] = all(x["labels_md5"] == y["labels_md5"] for x, y in zip(a["hist"], b["hist"]))
                if s == 42:
                    ra = sum(h["epoch_s"] for h in a["hist"][1:])
                    rb = sum(h["epoch_s"] for h in b["hist"][1:])
                    speed_ref[ds] = rb / ra
                    speed_tot[ds] = b["hist"][-1]["total_s"] / a["hist"][-1]["total_s"]
                    speed_later[ds] = [y["epoch_s"] / x["epoch_s"] for x, y in zip(a["hist"][2:], b["hist"][2:])]
            elif a:
                lack.append(s)
        if lack:
            missing(f"{ds} cr_nocert seeds {lack}")
        a, s_ = refine_json(ds, "cr", 42), refine_json(ds, "single", 42)
        if a and s_:
            single[ds] = dict(single=s_["hist"][-1]["skipped_pct"], per=a["hist"][-1]["skipped_pct"],
                              same=all(x["labels_md5"] == y["labels_md5"] for x, y in zip(a["hist"], s_["hist"])))
        elif a:
            missing(f"{ds} single_s42")
    SUMMARY.update(identical=ident, cert_speed_refine=speed_ref, cert_speed_total=speed_tot, single=single)
    if ident:
        setv("rq3.ident.count", sum(ident.values()))
        setv("rq3.ident.total", len(ident))
        setv("rq3.ident.seeds", len({k.split("|")[1] for k in ident}))
        if not all(ident.values()):
            print("WARNING: CertRefine with/without certificates differ on", [k for k, v in ident.items() if not v])
    # 批量 Lloyd（HSL）开/关证书：全部数据集 × 种子逐轮比较分配的 md5
    hi, hn = 0, 0
    for ds in DS:
        for s in L.SEEDS:
            ha, hb = refine_json(ds, "hsl", s), refine_json(ds, "hslc", s)
            if ha and hb:
                ha, hb = ha["hist"], hb["hist"]
                hn += 1
                hi += int(len(ha) == len(hb) and all(x["labels_md5"] == y["labels_md5"] for x, y in zip(ha, hb)))
    if hn:
        setv("rq3.ident.hsl.count", hi)
        setv("rq3.ident.hsl.total", hn)
        if hi != hn:
            print("WARNING: HSL with/without certificates differ in", hn - hi, "runs; the text claims identity in all runs")
    if speed_later:
        med = {ds: float(np.median(v)) for ds, v in speed_later.items()}
        SUMMARY["cert_speed_later_pass_median"] = med
        setv("rq3.certpass.min", min(med.values()), 1)
        setv("rq3.certpass.max", max(med.values()), 1)
    if speed_ref:
        setv("rq3.certspeed.min", min(speed_ref.values()), 1)
        setv("rq3.certspeed.max", max(speed_ref.values()), 1)
        setv("rq3.certtot.min", min(speed_tot.values()), 2)
        setv("rq3.certtot.max", max(speed_tot.values()), 2)
    if single:
        dss = list(single)
        ncol = 1 + len(dss)
        body = [rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Candidate distances skipped in pass 5 (\%)}}}} \\",
                "One bound per point & " + " & ".join(f"{single[ds]['single']:.2f}" for ds in dss) + r" \\",
                "$K$ bounds per point & " + " & ".join(f"{single[ds]['per']:.1f}" for ds in dss) + r" \\",
                r"\midrule",
                "Same partition & " + " & ".join("yes" if single[ds]["same"] else "no" for ds in dss) + r" \\"]
        write_gen("tab_single.tex", table(
            "table*", r"Skipped distances: one bound versus $K$ bounds per point.", "tab:single", "l" + "r" * len(dss),
            ["Bounds & " + " & ".join(L.DSNAME[ds] for ds in dss) + r" \\"], body))
        setv("rq3.single.max", max(v["single"] for v in single.values()), 1)
        setv("rq3.single.min", min(v["single"] for v in single.values()), 2)
        setv("rq3.single.per.min", min(v["per"] for v in single.values()), 0)
    # 每轮移动的点的比例（CertRefine，全部种子）；不动点：第 5 轮仍移动的比例
    mv1, mv5, mv5d = [], [], []
    for ds in DS:
        for s in L.SEEDS:
            e1, e5 = entry(ds, "cr", 1), entry(ds, "cr", 5)
            if e1 and e5 and s in e1 and s in e5:
                n_ = e1[s]["refine"]["n"]
                mv1.append(100.0 * e1[s]["hist"]["moves"] / n_)
                mv5.append(100.0 * e5[s]["hist"]["moves"] / n_)
                mv5d.append((100.0 * e5[s]["hist"]["moves"] / n_, ds))
    if mv1:
        setv("move.p1.min", min(mv1), 0)
        setv("move.p1.max", max(mv1), 0)
        setv("move.p5.min", min(mv5), 1)
        setv("move.p5.max", max(mv5), 1)
        setv("fixpt.p5.moves.min", min(mv5), 2)
        setv("fixpt.p5.moves.max", max(mv5), 2)
        setv("fixpt.p5.moves.ds.min", L.DSNAME[min(mv5d)[1]])
        setv("fixpt.p5.moves.ds.max", L.DSNAME[max(mv5d)[1]])
    mvd = {}
    for ds in DS:
        e1, e5 = entry(ds, "cr", 1), entry(ds, "cr", 5)
        if e1 and e5:
            s_ = [s for s in L.SEEDS if s in e1 and s in e5]
            if s_:
                mvd[ds] = (np.mean([100.0 * e1[s]["hist"]["moves"] / e1[s]["refine"]["n"] for s in s_]),
                           np.mean([100.0 * e5[s]["hist"]["moves"] / e5[s]["refine"]["n"] for s in s_]))
    if mvd:
        top = max(mvd, key=lambda d: mvd[d][1])
        setv("move.most.ds", L.DSNAME[top])
        setv("move.most.p1", mvd[top][0], 0)
        setv("move.most.p5", mvd[top][1], 1)
        SUMMARY["moves_pct_by_ds"] = mvd
        if V.get("rq3.skip.lowest.ds") not in (None, L.DSNAME[top]):
            print("WARNING: lowest skip rate on", V["rq3.skip.lowest.ds"], "but most moves on", L.DSNAME[top], "- check the RQ3 sentence")
    # H-5 构建时间的组成（全部种子均值）：分层版 / 初始分配（含候选图与状态初始化）/ 5 轮精修
    bd = {}
    for ds in DS:
        parts = []
        for s in L.SEEDS:
            r = refine_json(ds, "cr", s)
            if r and len(r["hist"]) > 5:
                t, h = r["timing"], r["hist"]
                hier_ = t.get("hier_s", 0.0)
                init_ = t.get("init_assign_s", 0.0) + t.get("graph0_s", 0.0) + t.get("setup_s", 0.0)
                parts.append((hier_, init_, h[5]["total_s"] - hier_ - init_, h[5]["total_s"]))
        if parts:
            a = np.mean(parts, axis=0)
            bd[ds] = dict(hier=a[0], init=a[1], passes=a[2], total=a[3], seeds=len(parts))
    SUMMARY["breakdown_cr5"] = bd
    if bd:
        share = {ds: {k: 100 * v[k] / v["total"] for k in ("hier", "init", "passes")} for ds, v in bd.items()}
        setv("bd.init.largest.count", sum(1 for v in share.values() if v["init"] >= max(v["hier"], v["passes"])))
        for k in ("init", "hier", "passes"):
            setv(f"bd.{k}.min", min(v[k] for v in share.values()), 0)
            setv(f"bd.{k}.max", max(v[k] for v in share.values()), 0)
        if "deep10m" in share:
            setv("bd.init.deep", share["deep10m"]["init"], 0)
        hl = [L.DSNAME[ds] for ds, v in share.items() if v["hier"] > v["init"]]
        setv("bd.hierlargest.list", " and ".join(hl) if hl else "none")
        setv("bd.seeds", min(v["seeds"] for v in bd.values()))
    # 提交校验拒绝率
    rej = []
    for ds in DS:
        for s in L.SEEDS:
            for e in range(1, 6):
                en = entry(ds, "cr", e)
                if en and s in en:
                    h = en[s]["hist"]
                    prop = h["moves"] + h["rejected"]
                    if prop:
                        rej.append(100.0 * h["rejected"] / prop)
    if rej:
        setv("rq3.reject.max", max(rej), 2)
    # 上限（ceiling）：第 5 轮每点仍要算的距离数、提交阶段的时间占比、判定阶段占总时间的比例（cr_s3/cr_s4：带 decide_s/commit_s 的重计时种子）
    ceil = {}
    for ds in DS:
        runs = [refine_json(ds, "cr", s) for s in L.SEEDS]
        runs = [r for r in runs if r and len(r["hist"]) > 5 and all("decide_s" in h for h in r["hist"][2:6])]
        if not runs:
            missing(f"{ds} cr_s3/cr_s4 with decide_s/commit_s (ceiling)")
            continue
        K = runs[0]["K"]
        dist5 = float(np.mean([(1 - r["hist"][5]["skipped_pct"] / 100) * K + 1 for r in runs]))
        commit = {e: float(np.mean([100 * r["hist"][e]["commit_s"] / (r["hist"][e]["decide_s"] + r["hist"][e]["commit_s"]) for r in runs]))
                  for e in range(2, 6)}
        dshare = float(np.mean([100 * sum(r["hist"][e]["decide_s"] for e in range(2, 6)) / r["hist"][5]["total_s"] for r in runs]))
        ceil[ds] = dict(dist5=dist5, commit=commit, decide_share=dshare, seeds=len(runs))
        setv(f"ceil.{ds}.p5.dist", dist5, 1)
        setv(f"ceil.{ds}.p2.commit", commit[2], 0)
        setv(f"ceil.{ds}.p5.commit", commit[5], 0)
        setv(f"ceil.{ds}.decide.share", dshare, 0)
    SUMMARY["ceiling"] = ceil
    if ceil:
        ng = [v["dist5"] for ds, v in ceil.items() if ds != "glove200"]
        if ng:
            setv("ceil.p5.dist.min", min(ng), 1)
            setv("ceil.p5.dist.max", max(ng), 1)
        if "glove200" in ceil:
            setv("ceil.p5.dist.glove", ceil["glove200"]["dist5"], 1)
        cm = [c for v in ceil.values() for c in v["commit"].values()]
        setv("ceil.commit.min", min(cm), 0)
        setv("ceil.commit.max", max(cm), 0)
        setv("ceil.decide.share.min", min(v["decide_share"] for v in ceil.values()), 0)
        setv("ceil.decide.share.max", max(v["decide_share"] for v in ceil.values()), 0)
        setv("ceil.seeds", min(v["seeds"] for v in ceil.values()))
        if "deep10m" in ceil:
            setv("ceil.deep.p2.commit", ceil["deep10m"]["commit"][2], 0)
            setv("ceil.deep.p5.commit", ceil["deep10m"]["commit"][5], 0)
    # 轮数扫描（seed 42 精确起步）：每轮之后的 recall
    ps = {}
    for ds in DS:
        row = {}
        for e in range(1, 6):
            en = entry(ds, "cr", e)
            if en and 42 in en and en[42]["scan"]:
                row[e] = 100 * en[42]["scan"]["recall@scan0.01"]
        if len(row) == 5:
            ps[ds] = row
            for e, v in row.items():
                setv(f"psweep.{ds}.p{e}", v, 2)
        else:
            missing(f"{ds} cr_s42 ep1-5 scan (pass sweep)")
    SUMMARY["psweep"] = ps
    if ps:
        setv("psweep.p2.gain.min", min(v[2] - v[1] for v in ps.values()), 2)
        setv("psweep.p2.gain.max", max(v[2] - v[1] for v in ps.values()), 2)
        setv("psweep.p5.vs.p2.max", max(v[5] - v[2] for v in ps.values()), 2)
        setv("psweep.p5.vs.p2.min", min(v[5] - v[2] for v in ps.values()), 2)
        setv("psweep.p5.vs.p1.min", min(v[5] - v[1] for v in ps.values()), 2)
        setv("psweep.p5.vs.p1.max", max(v[5] - v[1] for v in ps.values()), 2)
    # 内存（解析式）：x = n·d·4，bounds = n·K·4，state = n·20 字节
    for ds in ("deep10m", "msmarco"):
        r = refine_json(ds, "cr", 42)
        if r:
            setv(f"mem.{ds}.x", r["n"] * r["d"] * 4 / 1e9, 1)
            setv(f"mem.{ds}.bounds", r["n"] * r["K"] * 4 / 1e9, 1)
            setv(f"mem.{ds}.state", r["n"] * 20 / 1e9, 1)
            setv("mem.bounds.perpoint", r["K"] * 4)
            setv("mem.state.perpoint", 20)
        else:
            missing(f"{ds} cr_s42.refine.json (memory)")
    # 峰值内存（GNU time -v），若有
    for f in glob.glob(os.path.join(L.RES, "peakmem", "*.txt")):
        m = re.match(r"([a-z0-9]+)_(.+)\.txt$", os.path.basename(f))
        if not m:
            continue
        for line in open(f, errors="replace"):
            if "Maximum resident set size" in line:
                kb = float(line.strip().split()[-1])
                setv(f"mem.peak.{m.group(1)}.{m.group(2)}.gb", kb / 1e6, 1)
    # 确定性：不同线程数下逐轮 labels_md5 与 cr_s42 一致
    det = {}
    for ds in DS:
        base = refine_json(ds, "cr", 42)
        thr = {t: refine_json(ds, f"cr_thr{t}", 42) for t in (1, 2, 5, 10, 20)}
        if base and any(thr.values()):
            have = {t: r for t, r in thr.items() if r}
            det[ds] = all(len(r["hist"]) == len(base["hist"]) and all(x["labels_md5"] == y["labels_md5"] for x, y in zip(r["hist"], base["hist"]))
                          for r in have.values()) and len(have) == 5
            setv(f"sys.det.{ds}", det[ds])
            setv(f"sys.det.{ds}.threads", len(have))
    if det:
        setv("sys.det.count", sum(det.values()))
        setv("sys.det.total", len(det))
    # 图：(a) 各轮跳过比例；(b) 精修阶段时间 有/无证书
    if skip and speed_ref:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=FS(2.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
        style(a1)
        style(a2)
        for i, ds in enumerate(DS):
            if ds in skip:
                a1.plot(range(1, 6), skip[ds], marker=DSMARK[ds], ms=3.8, mec="white", mew=0.4, color=DSCOL[ds], label=L.DSNAME[ds])
        a1.set_xticks(range(1, 6))
        a1.set_xlabel("refinement pass")
        a1.set_ylabel("candidate distances skipped (%)")
        a1.set_ylim(-3, 122)
        a1.set_yticks(range(0, 101, 20))
        a1.set_title("(a) work removed by certificates", loc="left")
        a1.legend(*ds_legend(*a1.get_legend_handles_labels()), ncol=4, loc="upper center", fontsize=6.3,
                  columnspacing=0.9, handlelength=1.6, handletextpad=0.35)
        dss = [d for d in DS if d in speed_ref]
        ratios = [speed_ref[d] for d in dss]
        yy = np.arange(len(dss))[::-1]
        a2.axvline(1.0, color=AXIS, lw=0.8)
        a2.hlines(yy, 1.0, ratios, color=BLUE, lw=1.2)
        a2.plot(ratios, yy, "o", color=BLUE, ms=6, mec="white", mew=0.6)
        for yi, r in zip(yy, ratios):
            a2.text(r + 0.03, yi, f"{r:.2f}×", va="center", fontsize=6.5, color=INK2)
        a2.set_yticks(yy)
        a2.set_yticklabels([L.DSNAME[d] for d in dss])
        a2.set_xlim(0.9, max(ratios) * 1.2)
        a2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        a2.set_xlabel("time without / with certificates")
        a2.set_title("(b) speedup of the five passes", loc="left")
        fig.tight_layout(w_pad=1.5)
        fig.savefig(os.path.join(FIG, "fig_cert.pdf"))
        plt.close(fig)


# ---------------------------------------------------------------- 峰值内存表（rq3_cert 解析 results/peakmem/*.txt 之后）
MEM_ROWS = [("cr", "oursHfive"), ("cr_labels", "oursHL"), ("hslc_labels", "oursBL"), ("faiss_ivf", "faissivf"), ("hor", "horfive")]
MEM_DS = [("deep10m", "Deep-10M"), ("msmarco", "MS~MARCO")]


def mem_table():
    """峰值常驻内存（GNU time -v 的 Maximum resident set size，训练进程，seed 42）与解析式的界状态（n·K·4 字节，mem.<ds>.bounds）。
    注意：认证精修的被测进程（refine.py）从另一个未计量的 SuperKMeans 进程写出的分层质心/标签起步，HOR-5 的分层构建与迭代在同一个
    被测进程里；加载时短暂持有两份向量，峰值出现在加载阶段。界状态已包含在认证精修的运行里，不是额外的内存。
    单元格直接取 mem.peak.<ds>.<tag>.gb 键的值，与正文引用的键一致。"""
    if not any(f"mem.peak.{ds}.{tag}.gb" in V for ds, _ in MEM_DS for tag, _ in MEM_ROWS):
        missing("results/peakmem/*.txt (peak-memory table)")
        return
    body = [nm + " & " + " & ".join(V.get(f"mem.peak.{ds}.{tag}.gb", "--") for tag, _ in MEM_ROWS) + " & " + V.get(f"mem.{ds}.bounds", "--") + r" \\"
            for ds, nm in MEM_DS]
    write_gen("tab_mem.tex", table(
        "table*", "Peak resident memory (GB).", "tab:mem", "l" + "r" * (len(MEM_ROWS) + 1),
        ["Collection & " + " & ".join(tex(m) for _, m in MEM_ROWS) + r" & bound state$^{\dagger}$ \\"], body,
        note=r"Maximum resident set size of the training process (GNU time, seed 42). Every method loads the full collection. "
             r"$^{\dagger}$Analytic size of the certificates, included in the runs of \ours."))


# ---------------------------------------------------------------- RQ3：起步分配（3 个种子）/ 更新规则与判据
START_SEEDS = [42, 1, 2]
APPROX = [("cr_greedy", "greedy"), ("cr_ivfgemm", "ivfgemm"), ("cr_ann", "ann")]
# 起步表：数据集为列（同主表），三个行块；近似起步只有 seed 42（行名带 *）
START_ROWS = [("labels", "hierarchical labels"), ("snap", "snap"), ("greedy", r"greedy$^{\ast}$"), ("ivfgemm", r"IVF-based$^{\ast}$"),
              ("ann", r"ANN-based$^{\ast}$"), ("exact", "exact")]
START_BLOCKS = {"agree": r"Agreement with the exact assignment (\%)", "t": "Start time (s)", "r": r"Recall@10 (\%) after five passes"}


def rq3_start_and_ablation():
    cells, agree, snapag, loss, exbest, nseed = {}, [], [], [], 0, []   # cells: (块, 起步) -> {ds: 文本}
    for ds in DS:
        jl = refine_json(ds, "cr_labels", 42)
        if not jl or not refine_json(ds, "cr_snap", 42) or not refine_json(ds, "cr", 42):
            missing(f"{ds} cr_labels/cr_snap/cr seed 42 (start table)")
            continue
        # 一致率：标签起步来自 cr_labels_s42 + cr_labdg_s{1,2}（缺时用 cr_snap_s{1,2} 的 hier_labels_agree_exact）；snap 来自 cr_snap_s*
        ha = [jl["diag"]["hier_labels_agree_exact"]]
        sa, lack = [], []
        for s in START_SEEDS:
            jd = refine_json(ds, "cr_labdg", s) if s != 42 else None
            js = refine_json(ds, "cr_snap", s)
            if s != 42:
                src = jd or js
                if src and "hier_labels_agree_exact" in src.get("diag", {}):
                    ha.append(src["diag"]["hier_labels_agree_exact"])
                elif not jd:
                    lack.append(s)
            if js and "agree_exact" in js.get("diag", {}):
                sa.append(js["diag"]["agree_exact"])
        if lack:
            missing(f"{ds} cr_snap_s{lack} / cr_labdg_s{lack} (start table, seeds 1/2)")
        ha, sa = 100 * float(np.mean(ha)), 100 * float(np.mean(sa))
        agree.append(ha)
        snapag.append(sa)

        def tr(tag):
            ts, rs = [], []
            for s in START_SEEDS:
                j = refine_json(ds, tag, s)
                en = entry(ds, tag, 5)
                if j and en and s in en and en[s]["scan"]:
                    ts.append(j["timing"]["init_assign_s"] + j["timing"]["graph0_s"])
                    rs.append(100 * en[s]["scan"]["recall@scan0.01"])
            return (float(np.mean(ts)), float(np.mean(rs)), len(ts)) if ts else (np.nan, np.nan, 0)
        tl, ts_, tc = tr("cr_labels"), tr("cr_snap"), tr("cr")
        nseed.append(min(tl[2], ts_[2], tc[2]))
        loss.append(tc[1] - tl[1])
        exbest += int(tc[1] >= max(tl[1], ts_[1]))
        for st, a in (("labels", ha), ("snap", sa)):
            cells.setdefault(("agree", st), {})[ds] = f"{a:.1f}"
        for st, v in (("labels", tl), ("snap", ts_), ("exact", tc)):
            cells.setdefault(("t", st), {})[ds] = fmt(v[0], 2)
            cells.setdefault(("r", st), {})[ds] = fmt(v[1], 2)
        for nm, v in (("labels", tl), ("snap", ts_), ("exact", tc)):
            setv(f"start.{nm}.{ds}.t", v[0], 2)
            setv(f"start.{nm}.{ds}.r", v[1], 2)
        setv(f"start.labels.{ds}.agree", ha, 1)
        setv(f"start.snap.{ds}.agree", sa, 1)
    if agree:
        setv("rq3.hieragree.min", min(agree), 0)
        setv("rq3.hieragree.max", max(agree), 0)
        setv("rq3.hierdisagree.max", 100 - min(agree), 0)
        setv("rq3.hierdisagree.min", 100 - max(agree), 0)
        setv("rq3.snapagree.min", min(snapag), 0)
        setv("rq3.snapagree.max", max(snapag), 0)
        setv("rq3.labelloss.max", max(loss), 2)
        setv("rq3.labelloss.min", min(loss), 2)
        setv("rq3.exactbest.count", exbest)
        setv("start.seeds", min(nseed))
        setv("start.seeds.max", max(nseed))
    # 近似起步（seed 42）：greedy / IVF-GEMM / ANN
    arows = []
    for tag, nm in APPROX:
        vals = {}
        for ds in DS:
            j = refine_json(ds, tag, 42)
            en = entry(ds, tag, 5)
            if j and en and 42 in en and en[42]["scan"]:
                ag = 100 * j["diag"].get("agree_exact", np.nan)
                t = j["timing"]["init_assign_s"] + j["timing"]["graph0_s"]
                r = 100 * en[42]["scan"]["recall@scan0.01"]
                vals[ds] = (ag, t, r)
                setv(f"start.{nm}.{ds}.agree", ag, 1)
                setv(f"start.{nm}.{ds}.t", t, 2)
                setv(f"start.{nm}.{ds}.r", r, 2)
                cells.setdefault(("agree", nm), {})[ds] = f"{ag:.1f}"
                cells.setdefault(("t", nm), {})[ds] = fmt(t, 2)
                cells.setdefault(("r", nm), {})[ds] = f"{r:.2f}"
        if vals:
            for i, k in enumerate(("agree", "t", "r")):
                setv(f"start.{nm}.{k}.min", min(v[i] for v in vals.values()), 2)
                setv(f"start.{nm}.{k}.max", max(v[i] for v in vals.values()), 2)
            arows.append(nm)
        else:
            missing(f"{tag}_s42 (approximate start)")
    setv("start.approx.n", len(arows))
    grp = [(q, [lab] + [cells[(q, st)].get(ds, "--") for ds in DS]) for q in START_BLOCKS for st, lab in START_ROWS if (q, st) in cells]
    if grp:
        write_gen("tab_start.tex", table(
            "table", "Effect of the starting assignment.", "tab:start", "l" + "r" * len(DS),
            ["Start & " + " & ".join(L.DSNAME[d] for d in DS) + r" \\"], grouped_rows(grp, 1 + len(DS), START_BLOCKS),
            note=r"$^{\ast}$Seed 42 only." if arows else None))
    # 4 种变体（顺序/批量 × Hartigan/Lloyd）逐轮 recall（seed 42）
    var = [("cr", "sequential Hartigan (" + PRINTED["oursHfive"][:-2] + ")", BLUE, "-"), ("hb", "batch Hartigan", BLUE, "--"),
           ("sl", "sequential Lloyd", ORANGE, "-"), ("hslc", "batch Lloyd (" + PRINTED["oursBfive"][:-2] + ")", ORANGE, "--")]
    have = [ds for ds in DS if all(entry(ds, t, 5) and 42 in entry(ds, t, 5) for t, *_ in var)]
    first_gain, last_gain = [], []
    if have:
        fig, axes = plt.subplots(2, 4, figsize=FS(3.6))
        for ax, ds in zip(axes.ravel(), DS):
            style(ax)
            if ds not in have:
                ax.set_visible(False)
                continue
            h0 = entry(ds, "hier", None)
            for t, lab, col, ls in var:
                ys = []
                for e in range(1, 6):
                    en = entry(ds, t, e)
                    ys.append(100 * en[42]["scan"]["recall@scan0.01"] if en and 42 in en and en[42]["scan"] else np.nan)
                ax.plot(range(1, 6), ys, ls, marker="o", color=col, label=lab, ms=3.5)
            if h0 and 42 in h0 and h0[42]["scan"]:
                ax.axhline(100 * h0[42]["scan"]["recall@scan0.01"], color=NEUTRAL, lw=1, ls=":", label=PRINTED["hierskm"])
            ax.set_xticks(range(1, 6))
            ax.set_title(L.DSNAME[ds], loc="left")

            def rec(t, e):
                en = entry(ds, t, e)
                return en[42]["scan"]["recall@scan0.01"] if en and 42 in en and en[42]["scan"] else np.nan
            first_gain.append(100 * (rec("cr", 1) - rec("hslc", 1)))
            last_gain.append(100 * (rec("cr", 5) - rec("hslc", 5)))
        for ax in axes[:, 0]:
            ax.set_ylabel("recall@10 at 1% (%)")
        for ax in axes[-1]:
            ax.set_xlabel("pass")
        hh, ll = axes[0, 0].get_legend_handles_labels()
        top_legend(fig, hh, ll, 5)
        fig.savefig(os.path.join(FIG, "fig_ablation.pdf"))
        plt.close(fig)
    fg = [x for x in first_gain if not np.isnan(x)]
    lg = [x for x in last_gain if not np.isnan(x)]
    if fg:
        setv("rq3.hvsl.p1.min", min(fg), 2)
        setv("rq3.hvsl.p1.max", max(fg), 2)
        setv("rq3.hvsl.p5.min", min(lg), 2)
        setv("rq3.hvsl.p5.max", max(lg), 2)
    SUMMARY["h_vs_l"] = dict(pass1=first_gain, pass5=last_gain)
    lab1 = []
    for ds in DS:
        a, b = entry(ds, "cr_labels", 1), entry(ds, "hslc_labels", 1)
        if a and b and 42 in a and 42 in b and a[42]["scan"] and b[42]["scan"]:
            lab1.append(100 * (a[42]["scan"]["recall@scan0.01"] - b[42]["scan"]["recall@scan0.01"]))
    if lab1:
        setv("rq3.lab.hvsl.p1.min", min(lab1), 2)
        setv("rq3.lab.hvsl.p1.max", max(lab1), 2)
    SUMMARY["h_vs_l_labels_pass1"] = lab1


# ---------------------------------------------------------------- 额外基线表（补充材料）
EXTRA = [("bcdkm", None, "Batched CDKM"), ("bcdkm_samp", None, "Batched CDKM (30\\% sample)"),
         ("faiss_sampled", None, r"\faisssampled"),
         ("lloyd_exact", None, "SKM-Lloyd, no pruning"), ("sk_elkan", None, "scikit-learn Elkan"),
         ("sk_minibatch", None, r"\minibatch"), ("k2style", 5, r"\ktwostyle")]


def extra_table():
    """补充基线（正文 Table tab:extra）：上下两块（构建时间、召回），每格一个数；种子数单列（范围）。"""
    rows_t, rows_r = [], []
    for m, e, lab in EXTRA:
        ts, rs, ks, any_ = [], [], [], False
        for ds in DS:
            st_ = stat(ds, m, e)
            if st_ and isnum(st_["r"]):
                ts.append(tstr(st_["t"]))
                rs.append(f"{100 * st_['r']:.2f}")
                ks.append(st_["k"])
                any_ = True
            else:
                ts.append("--")
                rs.append("--")
        if any_:
            seeds = str(min(ks)) if min(ks) == max(ks) else f"{min(ks)}--{max(ks)}"
            rows_t.append(" & ".join([lab, seeds] + ts) + r" \\")
            rows_r.append(" & ".join([lab, seeds] + rs) + r" \\")
        elif m in ("sk_minibatch", "k2style"):
            missing(f"{m} runs (supplement table)")
    ncol = 2 + len(DS)
    body = [rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Build time (s)}}}} \\"] + rows_t + [r"\midrule",
            rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{Recall@10 (\%) at a 1\% scan budget}}}} \\"] + rows_r
    write_gen("tab_extra.tex", table(
        "table*", "Additional baselines.", "tab:extra", "lc" + "r" * len(DS),
        ["Method & Seeds & " + " & ".join(L.DSNAME[d] for d in DS) + r" \\"], body, tabcolsep="4pt",
        note="--: not run."))
    # Yahoo：剪枝是否是 FAISS 领先的原因
    for ds in DS:
        a, b, f = stat(ds, "lloyd", None), stat(ds, "lloyd_exact", None), stat(ds, "faiss", None)
        if a and b and f:
            setv(f"ds.{ds}.lloydexact", 100 * b["r"], 2)
            setv(f"ds.{ds}.lloydexact.t", b["t"], 0)
    # batched CDKM 的退化种子（某个列表远大于其余列表）：记录最严重的一次
    degen = []
    for ds in DS:
        en = entry(ds, "bcdkm", None)
        for s, run in (en or {}).items():
            sc = run["scan"]
            if sc and sc["cv"] > 1.5:
                degen.append((sc["cv"], ds, s, sc["max_list"], sc["recall@scan0.01"]))
    SUMMARY["bcdkm_degenerate"] = degen
    if degen:
        cv, ds, s, mx, r = max(degen)
        others = [x["scan"]["recall@scan0.01"] for ss, x in entry(ds, "bcdkm", None).items() if ss != s and x["scan"]]
        setv("bcdkm.degen.ds", L.DSNAME[ds])
        setv("bcdkm.degen.count", len(degen))
        setv("bcdkm.degen.cv", cv, 2)
        setv("bcdkm.degen.maxlist", f"{mx:,}")
        setv("bcdkm.degen.recall", 100 * r, 1)
        if others:
            setv("bcdkm.degen.others", 100 * float(np.mean(others)), 1)
    en = entry("gist", "bcdkm", None)
    if en:
        setv("bcdkm.gist.cv", float(np.mean([x["scan"]["cv"] for x in en.values() if x["scan"]])), 2)
    # MiniBatch（若有）：相对 Hier-SKM 的召回、相对 H-5 的时间
    mr, mt = [], []
    for ds in DS:
        mb, hi, c5 = stat(ds, "sk_minibatch", None), stat(ds, "hier", None), stat(ds, "cr", 5)
        if mb and isnum(mb["r"]) and hi and c5:
            mr.append(100 * (mb["r"] - hi["r"]))
            mt.append(mb["t"] / c5["t"])
    if mr:
        setv("mb.vs.hier.r.min", min(mr), 2)
        setv("mb.vs.hier.r.max", max(mr), 2)
        setv("mb.vs.cr5.t.min", min(mt), 2)
        setv("mb.vs.cr5.t.max", max(mt), 2)
        setv("mb.n", len(mr))
        V.setdefault("mb.pareto.count", "0")
    # k2style（R=1 的 B-L）相对 B-L
    kr, kt = [], []
    for ds in DS:
        k2, bl = stat(ds, "k2style", 5), stat(ds, "hslc_labels", 5)
        if k2 and isnum(k2["r"]) and bl:
            kr.append(100 * (k2["r"] - bl["r"]))
            kt.append(k2["t"] / bl["t"])
    if kr:
        setv("k2.vs.bl.r.min", min(kr), 2)
        setv("k2.vs.bl.r.max", max(kr), 2)
        setv("k2.vs.bl.t.min", min(kt), 2)
        setv("k2.vs.bl.t.max", max(kt), 2)
        setv("k2.n", len(kr))


# ---------------------------------------------------------------- 评测协议：固定 nprobe vs 等扫描
def c_of_ds(ds):
    en = entry(ds, "cr", 5)
    for run in (en or {}).values():
        return int(run["refine"]["c"])
    return 0


PROT_METH = [("hierskm", "hier"), ("oursHfive", "cr5"), ("hslskm", "hslskm"), ("horfive", "hor"), ("wsthree", "ws3"),
             ("skmdef", "skmdef"), ("faissivf", "faissivf"), ("faissfull", "faiss"), ("skmlloyd", "lloyd"), ("bcdkm", "bcdkm")]


def protocol():
    """固定 nprobe 与等扫描预算两种评测协议的对比（种子均值，稠密曲线）。
    固定 nprobe 取 round(c/100)：对完全均衡的划分恰好扫描 1%，因此两种协议在"理想"情况下一致。"""
    from scipy.stats import kendalltau
    out, taus, rev = {}, {}, {}
    for ds in DS:
        rows = {}
        for m, key in PROT_METH:
            en = entry(ds, "bcdkm", None) if m == "bcdkm" else entry(ds, *BY_MACRO[m])
            if not en:
                continue
            vals = []
            for s, run in en.items():
                sc = run["scan"]
                if not sc or len(sc["nprobes"]) < 50:
                    continue
                npf = max(1, int(round(c_of_ds(ds) / 100)))
                j = sc["nprobes"].index(npf) if npf in sc["nprobes"] else None
                if j is None:
                    continue
                vals.append((sc["recall"][j], sc["scan_frac"][j], sc["recall@scan0.01"], sc["cv"]))
            if vals:
                a = np.mean(vals, axis=0)
                rows[m] = dict(npf=a[0], fracf=a[1], eq=a[2], cv=a[3], k=len(vals))
        out[ds] = rows
        if len(rows) >= 4:
            labs = list(rows)
            taus[ds] = float(kendalltau([rows[x]["npf"] for x in labs], [rows[x]["eq"] for x in labs]).correlation)
            n_rev, n_pairs = 0, 0
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    a, b = rows[labs[i]], rows[labs[j]]
                    n_pairs += 1
                    n_rev += int((a["npf"] - b["npf"]) * (a["eq"] - b["eq"]) < 0)
            rev[ds] = (n_rev, n_pairs)
    SUMMARY["protocol"] = {ds: {PRINTED.get(m, m): v for m, v in r.items()} for ds, r in out.items()}
    SUMMARY["protocol_tau"] = taus
    SUMMARY["protocol_reversed"] = rev
    if taus:
        setv("prot.tau.min", min(taus.values()), 2)
        setv("prot.tau.max", max(taus.values()), 2)
        setv("prot.tau.lowds", L.DSNAME[min(taus, key=taus.get)])
        setv("prot.pairs", len(PROT_METH) * (len(PROT_METH) - 1) // 2)
        setv("prot.reversed.min", min(v[0] for v in rev.values()))
        setv("prot.reversed.max", max(v[0] for v in rev.values()))
        for ds, (a, b) in rev.items():
            setv(f"prot.{ds}.reversed", a)
            setv(f"prot.{ds}.pairs", b)
            setv(f"prot.{ds}.tau", taus[ds], 2)
    # 例子：GIST（种子均值）
    g = out.get("gist", {})
    if "skmdef" in g and "oursHfive" in g:
        c_g = c_of_ds("gist")
        setv("prot.gist.nprobe", int(round(c_g / 100)))
        for m, key in PROT_METH:
            if m in g:
                setv(f"prot.gist.{key}.np", 100 * g[m]["npf"], 1)
                setv(f"prot.gist.{key}.eq", 100 * g[m]["eq"], 1)
                setv(f"prot.gist.{key}.cv", g[m]["cv"], 2)
                setv(f"prot.gist.{key}.scan", 100 * g[m]["fracf"], 2)
        setv("prot.gist.scanratio", g["skmdef"]["fracf"] / g["oursHfive"]["fracf"], 1)
    if taus:
        # 两个并排的小表（顶端对齐）：左为各数据集的排名一致性，右为 GIST 上的例子（各量各占一列）
        left = [r"\begin{minipage}[t]{0.36\linewidth}", r"\centering", r"\textit{(a) Ranking agreement}\par\smallskip",
                r"\fittable{lrr}{", r"\toprule", r"Collection & Kendall's $\tau$ & reversed pairs \\", r"\midrule"]
        left += [f"{L.DSNAME[ds]} & {fmt(taus[ds], 2)} & {rev[ds][0]}/{rev[ds][1]}" + r" \\" for ds in DS if ds in taus]
        blocks = ["\n".join(left + [r"\bottomrule", r"}", r"\end{minipage}"])]
        gm = [(m, "Batched CDKM" if m == "bcdkm" else tex(m)) for m, key in PROT_METH if m in g]
        if gm and "prot.gist.nprobe" in V:
            right = [r"\begin{minipage}[t]{0.60\linewidth}", r"\centering", r"\textit{(b) GIST at a fixed \texttt{nprobe} and at an equal scan}\par\smallskip",
                     r"\fittable{lrrrr}{", r"\toprule",
                     r" & \multicolumn{2}{c}{\texttt{nprobe}$=$\V{prot.gist.nprobe}} & 1\% scan & \\",
                     r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}", r"Method & recall (\%) & scanned (\%) & recall (\%) & cv \\", r"\midrule"]
            right += [f"{lab} & {100 * g[m]['npf']:.1f} & {100 * g[m]['fracf']:.2f} & {100 * g[m]['eq']:.1f} & {g[m]['cv']:.2f}" + r" \\"
                      for m, lab in gm]
            blocks.append("\n".join(right + [r"\bottomrule", r"}", r"\end{minipage}"]))
        write_gen("tab_protocol.tex", "\n".join([
            r"\begin{table*}[pos=!tb]", r"\centering\small", caption_tex(r"Fixed \texttt{nprobe} versus equal scan."), r"\label{tab:protocol}",
            r"\setlength{\tabcolsep}{3pt}", r"\hfill".join(blocks), r"\end{table*}"]) + "\n")
        setv("prot.n", len(PROT_METH))


# ---------------------------------------------------------------- 吞吐（QPS）：同 recall 下相对 FAISS 的吞吐
def qps():
    """查询吞吐（seed 42，FAISS IndexIVFFlat，20 线程）：同 recall 下相对 FAISS-IVF（FAISS 的 IVF 默认配置）的吞吐。"""
    REF = "faiss_ivf_s42"
    show = [("cr_s42_ep5", "oursHfive", "certrefine5") + M_STYLE["oursHfive"],
            ("hsl_cpp_s42", "hslskm", "hslskm") + M_STYLE["hslskm"], ("hor_s42", "horfive", "hor") + M_STYLE["horfive"],
            ("skm_default_s42", "skmdef", "skmdefault") + M_STYLE["skmdef"], ("faiss_s42", "faissfull", "faissfull") + M_STYLE["faissfull"],
            ("lloyd_s42", "skmlloyd", "skmlloyd") + M_STYLE["skmlloyd"], ("hier_s42", "hierskm", "hierskm") + M_STYLE["hierskm"]]
    curves = {}
    for ds in DS:
        d = os.path.join(L.RES, ds)
        cs = {}
        for nm in [REF] + [x[0] for x in show]:
            p = os.path.join(d, nm + ".qps.json")
            if os.path.exists(p):
                cs[nm] = json.load(open(p))["curve"]
        if REF in cs and "cr_s42_ep5" in cs:
            curves[ds] = cs
        else:
            missing(f"{ds} qps curves (faiss_ivf_s42 / cr_s42_ep5)")

    def qps_at(cv, r):
        rr = np.array([x["recall"] for x in cv])
        qq = np.array([x["qps"] for x in cv])
        o = np.argsort(rr)
        rr, qq = rr[o], qq[o]
        if r < rr[0] or r > rr[-1]:
            return np.nan
        return float(np.exp(np.interp(r, rr, np.log(qq))))

    rel = {nm: {} for nm, *_ in show}
    for ds, cs in curves.items():
        en = entry(ds, "faiss_ivf", None)
        if not en or 42 not in en or not en[42]["scan"]:
            continue
        tgt = en[42]["scan"]["recall@scan0.01"]
        qf = qps_at(cs[REF], tgt)
        for nm, *_ in show:
            if nm in cs:
                rel[nm][ds] = qps_at(cs[nm], tgt) / qf
    SUMMARY["qps_rel_faissivf"] = rel
    for nm, m, key, *_ in show:
        v = [x for x in rel[nm].values() if not np.isnan(x)]
        if v:
            setv(f"qps.{key}.min", min(v), 2)
            setv(f"qps.{key}.max", max(v), 2)
            setv(f"qps.{m}.min", min(v), 2)
            setv(f"qps.{m}.max", max(v), 2)
    rh = []
    for ds, cs in curves.items():
        en = entry(ds, "faiss_ivf", None)
        if "hier_s42" in cs and en and 42 in en and en[42]["scan"]:
            tgt = en[42]["scan"]["recall@scan0.01"]
            rh.append(qps_at(cs["cr_s42_ep5"], tgt) / qps_at(cs["hier_s42"], tgt))
    rh = [x for x in rh if not np.isnan(x)]
    if rh:
        setv("qps.vshier.min", min(rh), 2)
        setv("qps.vshier.max", max(rh), 2)
    if not curves:
        return
    fig, axes = plt.subplots(2, 4, figsize=FS(3.8))
    for ax, ds in zip(axes.ravel(), DS):
        style(ax)
        if ds not in curves:
            ax.set_visible(False)
            continue
        cf = curves[ds][REF]
        tgt = 100 * entry(ds, "faiss_ivf", None)[42]["scan"]["recall@scan0.01"]
        ax.axhline(1.0, color=TEAL, lw=1.2, label=PRINTED["faissivf"] + " (reference)")
        for nm, m, key, col, ls in show:
            if nm not in curves[ds]:
                continue
            cv = curves[ds][nm]
            lo_r = max(min(x["recall"] for x in cv), min(x["recall"] for x in cf), tgt / 100 - 0.15)
            hi_r = min(max(x["recall"] for x in cv), max(x["recall"] for x in cf))
            if hi_r <= lo_r:
                continue
            grid = np.linspace(lo_r, hi_r, 60)
            ax.plot(100 * grid, [qps_at(cv, r) / qps_at(cf, r) for r in grid], ls=ls, color=col, lw=1.2, label=PRINTED[m])
        ax.axvline(tgt, color=AXIS, lw=0.7, ls=(0, (2, 2)))
        ax.set_title(L.DSNAME[ds], loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    for ax in axes[:, 0]:
        ax.set_ylabel("QPS relative to FAISS-IVF")
    for ax in axes[-1]:
        ax.set_xlabel("recall@10 (%)")
    hh, ll = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, hh, ll, 4)
    fig.savefig(os.path.join(FIG, "fig_qps.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- K 扫描（3 个种子）与 c 扫描（SIFT，3 个种子）
SENS_SEEDS = [42, 1, 2]
CS = [1000, 2000, 4000, 8000]
CSCALE_ROWS = ["hierskm", "oursHL", "oursHone", "oursHfive", "oursBL", "oursBfive", "hslskm", "horone", "horfive", "skmdef", "faissivf", "faissfull"]


def agg3(en, key="recall@scan0.01"):
    """种子 [42, 1, 2] 上的均值/SD（缺的种子跳过）：返回 dict(t, t_sd, r, r_sd, k)。"""
    if not en:
        return None
    tm, ts, vm, vs, k = L.agg(en, key, seeds=SENS_SEEDS)
    return dict(t=tm, t_sd=ts, r=vm, r_sd=vs, k=k) if k else None


def pm(m, sd, nd=2):
    return f"{fmt(m, nd)}" + ("" if sd is None or np.isnan(sd) else rf"$\pm${sd:.{nd}f}")


def ksweep_cscale():
    ks = [16, 32, 64, 128, 256]
    res, kseeds = {}, []
    for ds in DS:
        row, lack = {}, {}
        for K in ks:
            tag = "cr" if K == 64 else f"cr_K{K}"
            en = entry(ds, tag, 5)
            s_ = agg3(en)
            if s_:
                sk = [s for s in SENS_SEEDS if s in en and en[s]["scan"]]
                skip_ = float(np.mean([en[s]["hist"]["skipped_pct"] for s in sk]))
                row[K] = dict(r=s_["r"], r_sd=s_["r_sd"], t=s_["t"], t_sd=s_["t_sd"], skip=skip_, k=len(sk))
                if [s for s in SENS_SEEDS if s not in sk]:
                    lack[tag] = [s for s in SENS_SEEDS if s not in sk]
            else:
                lack[tag] = "all"
        if lack:
            missing(f"{ds} K sweep seeds: " + ", ".join(f"{t} {v}" for t, v in lack.items()))
        if len(row) == len(ks):
            res[ds] = row
            kseeds.append(min(v["k"] for v in row.values()))
    SUMMARY["ksweep"] = res
    if res:
        setv("k.seeds", min(kseeds))
        g256 = [100 * (v[256]["r"] - v[64]["r"]) for v in res.values()]
        g16 = [100 * (v[16]["r"] - v[64]["r"]) for v in res.values()]
        setv("k.gain256.min", min(g256), 2)
        setv("k.gain256.max", max(g256), 2)
        setv("k.loss16.min", min(g16), 2)
        setv("k.loss16.max", max(g16), 2)
        setv("k.loss16.worst", -min(g16), 2)   # K=16 相对 K=64 的最大损失（正数）
        best = max(res, key=lambda d: res[d][256]["r"] - res[d][64]["r"])
        setv("k.gain256.best.ds", L.DSNAME[best])
        g128 = [100 * (v[128]["r"] - v[64]["r"]) for v in res.values()]
        setv("k.gain128.min", min(g128), 2)
        setv("k.gain128.max", max(g128), 2)
        t128 = [v[128]["t"] / v[64]["t"] for v in res.values()]
        setv("k.time128.min", min(t128), 1)
        setv("k.time128.max", max(t128), 1)
        tr = [v[256]["t"] / v[64]["t"] for v in res.values()]
        setv("k.time256.min", min(tr), 1)
        setv("k.time256.max", max(tr), 1)
    ps = SUMMARY.get("psweep", {})
    if res or ps:
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=FS(2.4))
        for a in (a1, a2, a3):
            style(a)
        for i, ds in enumerate(DS):
            if ds in res:
                a1.errorbar(ks, [100 * (res[ds][K]["r"] - res[ds][64]["r"]) for K in ks],
                            yerr=[0 if np.isnan(res[ds][K]["r_sd"]) else 100 * res[ds][K]["r_sd"] for K in ks],
                            marker=DSMARK[ds], ms=3.8, mec="white", mew=0.4, color=DSCOL[ds], label=L.DSNAME[ds], elinewidth=0.6, capsize=0)
                a2.plot(ks, [res[ds][K]["t"] for K in ks], marker=DSMARK[ds], ms=3.8, mec="white", mew=0.4, color=DSCOL[ds], label=L.DSNAME[ds])
            if ds in ps:
                a3.plot(range(1, 6), [ps[ds][e] - ps[ds][1] for e in range(1, 6)], marker=DSMARK[ds], ms=3.8, mec="white", mew=0.4,
                        color=DSCOL[ds], label=L.DSNAME[ds])
        for a in (a1, a2):
            a.set_xscale("log", base=2)
            a.set_xticks(ks)
            a.set_xticklabels([str(k) for k in ks])
            a.set_xlabel("candidates $K$")
        a1.axhline(0, color=AXIS, lw=0.8)
        a1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        a1.set_ylabel("recall change vs. $K$=64 (points)")
        a1.set_title("(a) recall@10 at 1% scan, 5 passes", loc="left")
        a2.set_yscale("log")
        a2.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a2.set_ylabel("build time (s, log)")
        a2.set_title("(b) build time", loc="left")
        a3.set_xticks(range(1, 6))
        a3.set_xlabel("pass")
        a3.set_ylabel("recall gain over pass 1 (points)")
        a3.set_title("(c) passes (seed 42)", loc="left")
        hh, ll = ds_legend(*(a1 if res else a3).get_legend_handles_labels())
        fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.5)
        fig.legend(hh, ll, loc="upper center", ncol=len(ll), bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handletextpad=0.4)
        fig.savefig(os.path.join(FIG, "fig_ksweep.pdf"))
        plt.close(fig)
    # c 扫描（SIFT）：种子 [42, 1, 2]
    tab, n_sift, cseeds = {}, None, []
    for c in CS:
        dname = "sift" if c == 4000 else f"sift_c{c}"
        if not os.path.isdir(os.path.join(L.RES, dname)):
            missing(f"{dname} (c-scale)")
            continue
        rr = L.runs(dname)
        lack = {}
        for m in CSCALE_ROWS:
            en = rr.get(BY_MACRO[m])
            s_ = agg3(en)
            if s_:
                tab.setdefault(m, {})[c] = s_
                cseeds.append(s_["k"])
                if s_["k"] < len(SENS_SEEDS):
                    lack[BY_MACRO[m][0]] = [s for s in SENS_SEEDS if s not in en]
            else:
                lack[BY_MACRO[m][0]] = "all"
        if lack:
            missing(f"{dname} seeds: " + ", ".join(f"{t} {v}" for t, v in lack.items()))
        if n_sift is None:
            for en in rr.values():
                for run in en.values():
                    if "refine" in run:
                        n_sift = run["refine"]["n"]
                        break
                if n_sift:
                    break
    SUMMARY["cscale"] = {PRINTED[m]: {str(c): (v["t"], v["r"], v["k"]) for c, v in d.items()} for m, d in tab.items()}
    if not tab:
        return
    setv("c.seeds", min(cseeds))
    setv("c.seeds.max", max(cseeds))
    body = []
    for m in CSCALE_ROWS:
        if m in tab:
            cells = [pm(tab[m][c]["t"], tab[m][c]["t_sd"], 1) + " / " + pm(100 * tab[m][c]["r"], 100 * tab[m][c]["r_sd"], 2) if c in tab[m] else "--" for c in CS]
            body.append(tex(m) + " & " + " & ".join(cells) + r" \\")
    hdr = [rf" & \multicolumn{{{len(CS)}}}{{c}}{{build time (s) / recall@10 (\%), mean$\pm$SD}} \\", rf"\cmidrule(lr){{2-{1 + len(CS)}}}",
           "Method & " + " & ".join(f"$c={c}$" for c in CS) + r" \\"]
    if n_sift:
        hdr.append("$c/n$ (\\%) & " + " & ".join(f"{100 * c / n_sift:.2f}" for c in CS) + r" \\")
    write_gen("tab_cscale.tex", table(
        "table", r"Effect of the number of lists $c$ on SIFT.", "tab:cscale", "l" + "c" * len(CS), hdr, body))
    best_cnt, cheapest, cheapest_cert, fastest, vs_hor = 0, {}, {}, {}, []
    ours = ("oursHL", "oursHone", "oursHfive", "oursBL", "oursBfive")
    refine_set = ("oursHL", "oursHone", "oursHfive", "oursBfive", "horfive")
    for c in CS:
        cand = {m: tab[m][c] for m in tab if c in tab[m]}
        if cand and max(cand, key=lambda k: cand[k]["r"]) == "oursHfive":
            best_cnt += 1
        nh = {m: v for m, v in cand.items() if m != "hierskm"}
        if nh:
            cheapest[c] = min(nh, key=lambda k: nh[k]["t"])
            setv(f"c.cheapest.{c}.name", tex(cheapest[c]))
        oc = {m: v for m, v in cand.items() if m in ours}
        if oc:
            cheapest_cert[c] = min(oc, key=lambda k: oc[k]["t"])
            setv(f"c.cheapest.cert.{c}.name", tex(cheapest_cert[c]))
        ref = {m: v for m, v in cand.items() if m in refine_set}
        if len(ref) == len(refine_set):
            fastest[c] = min(ref, key=lambda k: ref[k]["t"])
        if "oursHfive" in cand and "horfive" in cand:
            vs_hor.append(cand["horfive"]["t"] / cand["oursHfive"]["t"])
    setv("c.cr5.best.count", best_cnt)
    SUMMARY["cscale_cheapest"] = {str(c): PRINTED[v] for c, v in cheapest.items()}
    SUMMARY["cscale_fastest"] = {str(c): PRINTED[v] for c, v in fastest.items()}

    def span(v):
        if v == CS:
            return "every $c$"
        if v[-1] == CS[-1]:
            return f"$c\\ge {v[0]}$"
        if v[0] == CS[0]:
            return f"$c\\le {v[-1]}$" if len(v) > 1 else f"$c={v[0]}$"
        return f"${v[0]}\\le c\\le {v[-1]}$" if len(v) > 1 else f"$c={v[0]}$"

    def phrase(d):
        groups = []
        for c in CS:
            if c not in d:
                continue
            if groups and groups[-1][0] == d[c]:
                groups[-1][1].append(c)
            else:
                groups.append([d[c], [c]])
        return " and ".join(f"{tex(m)} at {span(v)}" for m, v in groups)
    if len(cheapest) == len(CS):
        setv("c.cheapest.phrase", phrase(cheapest))
    if len(fastest) == len(CS):
        setv("c.fastest.phrase", phrase(fastest))
    if vs_hor:
        setv("c.hor.over.cr5.min", min(vs_hor), 1)
        setv("c.hor.over.cr5.max", max(vs_hor), 1)
    # H-5 时间的组成随 c 的变化（seed 42）：精修五轮基本不变，增长来自起步分配
    parts = {}
    for c in CS:
        r = refine_json("sift" if c == 4000 else f"sift_c{c}", "cr", 42)
        if r:
            parts[c] = (r["timing"]["init_assign_s"], sum(h["epoch_s"] for h in r["hist"][1:]))
    if len(parts) == len(CS):
        setv("c.cr5.init.min", min(v[0] for v in parts.values()), 1)
        setv("c.cr5.init.max", max(v[0] for v in parts.values()), 1)
        setv("c.cr5.passes.min", min(v[1] for v in parts.values()), 1)
        setv("c.cr5.passes.max", max(v[1] for v in parts.values()), 1)
    for m, key in (("oursHfive", "cr5"), ("horfive", "hor"), ("faissfull", "faiss"), ("oursHL", "crl"), ("oursBL", "bl"), ("faissivf", "fivf")):
        if m in tab and all(c in tab[m] for c in (CS[0], CS[-1])):
            setv(f"c.{key}.growth", tab[m][CS[-1]]["t"] / tab[m][CS[0]]["t"], 1)
        if m in tab and all(c in tab[m] for c in CS):
            setv(f"c.{key}.t.min", min(tab[m][c]["t"] for c in CS), 1)
            setv(f"c.{key}.t.max", max(tab[m][c]["t"] for c in CS), 1 if key != "faiss" else 0)


# ---------------------------------------------------------------- MS MARCO
MS_ROWS = [("hier", "hierskm", "hier"), ("faiss_2lc", "faisstwol", "hier"), ("cr_labels_ep5", "oursHL", "ours"), ("cr_ep1", "oursHone", "ours"), ("cr_ep5", "oursHfive", "ours"),
           ("hslc_labels_ep5", "oursBL", "ours"), ("hslc_ep5", "oursBfive", "ours"), ("hsl_cpp", "hslskm", "refine"), ("hor1", "horone", "refine"),
           ("hor2", "hortwo", "refine"), ("hor", "horfive", "refine"), ("warm1", "wsone", "refine"), ("warm2", "wstwo", "refine"),
           ("warm3", "wsthree", "refine"), ("skm_default", "skmdef", "sampled"), ("faiss_ivf", "faissivf", "sampled"), ("faiss", "faissfull", "full")]
_i5 = [r[1] for r in MS_ROWS].index("oursHfive") + 1
_iff = [r[1] for r in MS_ROWS].index("faissfull")
MS_SUPP = MS_ROWS[:_i5] + [("sl_ep5", "slfive", "ours")] + MS_ROWS[_i5:_iff] + [("faiss_sampled", "faisssampled", "sampled")] + MS_ROWS[_iff:]
MS_ENC_S = 4420911 / 1464.0   # 编码耗时（每个分片，logs/enc0.log 的速率；两个分片在两块 GPU 上并行）；not timed under the protocol
MET = ["mrr10", "ndcg10", "recall100"]
B_LO = 0.0025


def msmarco():
    """MS MARCO 端到端检索：GPU 稠密评测（eval_ir_gpu.py）的结果；两层自助法（种子 × 查询）；
    相对 H-5 的显著性（每列对表中其余全部行做 Holm）与等价标记；第二参照 FAISS-full 的等价检验。"""
    d = os.path.join(L.RES, "msmarco")
    meta_p = os.path.join(HERE, "..", "data_msmarco", "msmarco_meta.json")
    exact = {}
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        setv("ms.n", f"{meta['n']:,}")
        setv("ms.c", f"{meta['c']:,}")
        setv("ms.d", meta["d"])
        exact = meta["exact_search"]
        for k, v in exact.items():
            setv(f"ms.exact.{k}", 100 * v, 1)
    else:
        missing("data_msmarco/msmarco_meta.json")
    setv("ms.enc.s", MS_ENC_S, 0)
    if not os.path.isdir(d):
        missing("msmarco results")
        return
    fa = L.assign_s(d)
    rows = {}
    for f in glob.glob(os.path.join(d, "*.ir.json")):
        nm = os.path.basename(f)[:-len(".ir.json")]
        m = re.match(r"(.+?)_s(\d+)(?:_ep(\d+))?$", nm)
        if not m:
            continue
        r = json.load(open(f))
        if r.get("evaluator") != "gpu-dense-ir-v1" or not os.path.exists(os.path.join(d, nm + ".ir_pqb.npz")):
            continue
        meth, seed, ep = m.group(1), int(m.group(2)), m.group(3)
        key = meth + (f"_ep{ep}" if ep else "")
        if ep:
            rj = os.path.join(d, f"{meth}_s{seed}.refine.json")
            if not os.path.exists(rj):
                continue
            t = json.load(open(rj))["hist"][int(ep)]["total_s"]
        else:
            lg = os.path.join(d, f"{meth}_s{seed}.log")
            if not os.path.exists(lg):
                continue
            t = L.train_s(lg)
            if meth == "hsl_cpp":
                t += L.train_s(os.path.join(d, f"hier_s{seed}.log"))
        rows.setdefault(key, []).append(dict(t=t + fa, t_train=t, seed=seed, name=nm, cv=r["cv"], **{k: r[k] for k in r if "@scan" in k}))
    SUMMARY["msmarco"] = rows
    lack = [key for key, m, g in MS_SUPP if key not in rows]
    if lack:
        missing("msmarco rows " + ", ".join(lack))
    short = {key: [s for s in L.SEEDS if s not in {x["seed"] for x in rows[key]}] for key, m, g in MS_SUPP if key in rows}
    if any(short.values()):
        missing("msmarco seeds " + "; ".join(f"{key}: {v}" for key, v in short.items() if v))
    if "cr_ep5" not in rows:
        missing("msmarco cr_ep5 (reference)")
        return
    _pq = {}

    def pqs(key, mtr, b):
        if (key, mtr, b) not in _pq:
            mats = []
            for x in sorted(rows[key], key=lambda z: z["seed"]):
                arr = np.load(os.path.join(d, x["name"] + ".ir_pqb.npz"))[mtr]
                mats.append(arr[:, L.BUDGETS.index(b)].astype(np.float64))
            _pq[(key, mtr, b)] = np.vstack(mats)
        return _pq[(key, mtr, b)]

    main_present = [(key, m, g) for key, m, g in MS_ROWS if key in rows]
    cis, eq2 = {}, {}
    fam_ms = set()
    for mtr in MET + ["ann_recall10"]:
        for b in (B_LO, L.B1):
            ref = pqs("cr_ep5", mtr, b)
            keys, bts = [], []
            for key, m, g in main_present:
                if key == "cr_ep5":
                    continue
                bts.append(L.boot_diff(ref, pqs(key, mtr, b), n_boot=N_BOOT, delta=DELTA))   # H-5 减对照
                keys.append(key)
            fam_ms.add(len(keys))
            for key, pa, bt in zip(keys, L.holm([bt["p"] for bt in bts]), bts):
                cis[(key, mtr, b)] = dict(bt, p_holm=float(pa), sig=bool(pa < 0.05))
                SUMMARY.setdefault("msmarco_sig", {})[f"{key}|{mtr}|{b}"] = dict(
                    cr5_minus_pts=100 * bt["diff"], ci95=[100 * bt["lo95"], 100 * bt["hi95"]], ci90=[100 * bt["lo90"], 100 * bt["hi90"]],
                    p_holm=float(pa), equiv=bt["equiv"], family=len(keys))
            # 补充表里的行（SL、FAISS-sampled）只做单独的检验，不进 Holm 族
            for key, m, g in MS_SUPP:
                if key in rows and key not in keys and key != "cr_ep5":
                    bt = L.boot_diff(ref, pqs(key, mtr, b), n_boot=N_BOOT, delta=DELTA)
                    cis[(key, mtr, b)] = dict(bt, p_holm=float("nan"), sig=False)
            if "faiss" in rows and mtr in MET:
                ref2 = pqs("faiss", mtr, b)
                for key, m, g in MS_SUPP:
                    if key in rows and key != "faiss":
                        eq2[(key, mtr, b)] = L.boot_diff(pqs(key, mtr, b), ref2, n_boot=N_BOOT, delta=DELTA)   # 对照减 FAISS-full
    if fam_ms:
        setv("fam.ms", max(fam_ms))

    def cell(key, mtr, b, v, with_marks=True):
        c = cis.get((key, mtr, b))
        txt = f"{v:.2f}"
        if not with_marks or not c:
            return txt
        sup = ("\\blacktriangle" if c["diff"] < 0 else "\\triangledown") if c["sig"] else ""
        sup += "=" if c["equiv"] else ""
        return txt + (r"$^{" + sup + "}$" if sup else "")

    def emit(name, rowset, caption, label, with_seeds, note):
        hdr = [r" & " + ("& " if with_seeds else "") + r"& \multicolumn{2}{c}{MRR@10} & \multicolumn{2}{c}{nDCG@10} & \multicolumn{2}{c}{Recall@100} & \multicolumn{2}{c}{ANN recall@10} \\",
               (r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}" if with_seeds else
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}"),
               r"Method & " + ("Seeds & " if with_seeds else "") + r"Time (s) & 0.25\% & 1\% & 0.25\% & 1\% & 0.25\% & 1\% & 0.25\% & 1\% \\"]
        rws = []
        for key, m, g in rowset:
            if key not in rows:
                continue
            rr = rows[key]
            mean = lambda k: float(np.mean([x[k] for x in rr]))
            t = mean("t")
            cells = [cell(key, mtr, b, 100 * mean(f"{mtr}@scan{b}"), with_marks=(key, m, g) in main_present) for mtr in MET + ["ann_recall10"] for b in (B_LO, L.B1)]
            rws.append((g, [tex(m)] + ([str(len(rr))] if with_seeds else []) + [tstr(t)] + cells))
        ncol = 11 if with_seeds else 10
        write_gen(name, table("table*", caption, label, "l" + "r" * (ncol - 1), hdr, grouped_rows(rws, ncol), tabcolsep="3pt", note=note))
    extra_ms = [(key, m) for key, m, g in MS_SUPP if key in rows and (key, m, g) not in MS_ROWS]
    seeds_txt = lambda key: "" if len(rows[key]) == max(len(v) for v in rows.values()) else f" ({len(rows[key])} seeds)"
    emit("tab_msmarco.tex", MS_SUPP, "Passage retrieval on MS~MARCO.", "tab:msmarco", False,
         MARKS_H5 + ("; " + " and ".join(tex(m) + seeds_txt(key) for key, m in extra_ms) + ": not tested" if extra_ms else "") + ".")
    # 键：宏名 + 旧短名（key 去下划线）
    tmean = lambda k: float(np.mean([x["t"] for x in rows[k]]))
    seeds_n = []
    for key, m, g in MS_SUPP:
        if key not in rows:
            continue
        rr = rows[key]
        mean = lambda k: float(np.mean([x[k] for x in rr]))
        t = mean("t")
        if (key, m, g) in main_present:
            seeds_n.append(len(rr))
        for nm in (m, key.replace("_", "")):
            setv(f"ms.{nm}.t", t, 0 if t >= 100 else 1)
            setv(f"ms.{nm}.k", len(rr))
            setv(f"ms.k.{nm}", len(rr))
            for mtr in MET + ["ann_recall10"]:
                setv(f"ms.{nm}.{mtr.replace('_', '')}", 100 * mean(f"{mtr}@scan0.01"), 2)
                setv(f"ms.{nm}.{mtr.replace('_', '')}.q25", 100 * mean(f"{mtr}@scan{B_LO}"), 2)
            for mtr in MET + ["ann_recall10"]:
                for b, tag in ((L.B1, ""), (B_LO, ".q25")):
                    c = cis.get((key, mtr, b))
                    if c:
                        setv(f"ms.ci.{nm}.{mtr.replace('_', '')}{tag}.lo", 100 * c["lo95"], 2)
                        setv(f"ms.ci.{nm}.{mtr.replace('_', '')}{tag}.hi", 100 * c["hi95"], 2)
                        setv(f"ms.diff.{nm}.{mtr.replace('_', '')}{tag}", 100 * c["diff"], 2)
                        setv(f"ms.eq.{nm}.{mtr.replace('_', '')}{tag}", c["equiv"])
                        if not np.isnan(c["p_holm"]):
                            setv(f"ms.p.{nm}.{mtr.replace('_', '')}{tag}", c["p_holm"], 3)
                            setv(f"ms.sig.{nm}.{mtr.replace('_', '')}{tag}", c["sig"])
                    c2 = eq2.get((key, mtr, b))
                    if c2:
                        setv(f"ms.ci2.{nm}.{mtr.replace('_', '')}{tag}.lo", 100 * c2["lo95"], 2)
                        setv(f"ms.ci2.{nm}.{mtr.replace('_', '')}{tag}.hi", 100 * c2["hi95"], 2)
                        setv(f"ms.diff2.{nm}.{mtr.replace('_', '')}{tag}", 100 * c2["diff"], 2)
                        setv(f"ms.eq2.{nm}.{mtr.replace('_', '')}{tag}", c2["equiv"])
            if exact:
                for mtr in MET:
                    setv(f"ms.exact.gap.{nm}.{mtr}", 100 * (exact[mtr] - mean(f"{mtr}@scan0.01")), 2)
            c = cis.get((key, "mrr10", L.B1))
            if c:
                setv(f"ms.eq.delta025.{nm}.mrr10", bool(c["lo90"] > -DELTA / 2 and c["hi90"] < DELTA / 2))
    if seeds_n:
        setv("ms.seeds.min", min(seeds_n))
        setv("ms.seeds.max", max(seeds_n))
    # 汇总：等价性（对 H-5）、最便宜的等价行（对 FAISS-full）、与精确检索的差距、±0.25 的敏感性
    refined = [(key, m) for key, m, g in MS_ROWS if g in ("ours", "refine", "full") and key in rows and key != "cr_ep5"]
    for mtr in ("mrr10", "ndcg10"):
        noteq = [m for key, m in refined if not cis[(key, mtr, L.B1)]["equiv"]]
        setv(f"ms.eqall.{mtr}", not noteq)
        setv(f"ms.noteq.{mtr}.list", ", ".join(tex(m) for m in noteq) or "none")
        noteq_ex = [m for key, m in refined if not cis[(key, mtr, L.B1)]["equiv"] and m not in ("oursHL", "oursBL")]
        setv(f"ms.eqall.exact.{mtr}", not noteq_ex)
    gaps = [float(V[f"ms.exact.gap.{m}.mrr10"].replace(r"\ensuremath{-}", "-")) for key, m, g in MS_ROWS
            if g in ("ours", "refine", "full") and key in rows and exact]
    if gaps:
        setv("ms.exact.gap.min", min(gaps), 2)
        setv("ms.exact.gap.max", max(gaps), 2)
    flip = [m for key, m, g in MS_ROWS if key in rows and key != "cr_ep5" and cis[(key, "mrr10", L.B1)]["equiv"] and V.get(f"ms.eq.delta025.{m}.mrr10") == "no"]
    setv("ms.eq.delta025.flip.list", ", ".join(tex(m) for m in flip) or "none")
    setv("ms.eq.delta025.flip.count", len(flip))
    if "faiss" in rows:
        cand = sorted((tmean(key), m) for key, m, g in MS_ROWS if key in rows and key != "faiss"
                      and eq2[(key, "mrr10", L.B1)]["equiv"] and eq2[(key, "ndcg10", L.B1)]["equiv"])[:3]
        setv("ms.cheapest.n", len(cand))
        if cand:
            setv("ms.cheapest.list", ", ".join(tex(m) for _, m in cand))
            setv("ms.cheapest.t.min", min(t for t, _ in cand), 0)
            setv("ms.cheapest.t.max", max(t for t, _ in cand), 0)
    # 显著性列表、速度比、编码时间占比
    for mtr in MET + ["ann_recall10"]:
        for b, tag in ((L.B1, ""), (B_LO, ".q25")):
            setv(f"ms.sig.{mtr.replace('_', '')}{tag}.lower", ", ".join(tex(m) for key, m, g in main_present if cis.get((key, mtr, b), {}).get("sig") and cis[(key, mtr, b)]["diff"] > 0) or "none")
            setv(f"ms.sig.{mtr.replace('_', '')}{tag}.higher", ", ".join(tex(m) for key, m, g in main_present if cis.get((key, mtr, b), {}).get("sig") and cis[(key, mtr, b)]["diff"] < 0) or "none")
    for our in ("cr_ep5", "cr_labels_ep5", "cr_ep1", "hslc_labels_ep5"):
        for rv in ("faiss", "faiss_ivf", "skm_default", "hsl_cpp", "hor", "warm3", "hier"):
            if our in rows and rv in rows:
                setv(f"ms.speed.{rv.replace('_', '')}.{our.replace('_', '')}", tmean(rv) / tmean(our), 1)
                setv(f"ms.sp.{dict(MS_ROWS_M)[our]}.{dict(MS_ROWS_M)[rv]}", tmean(rv) / tmean(our), 1)
    shares = []
    for key, m in (("cr_ep5", "oursHfive"), ("cr_labels_ep5", "oursHL"), ("hslc_labels_ep5", "oursBL"), ("faiss_ivf", "faissivf"), ("faiss", "faissfull")):
        if key in rows:
            sh = 100 * tmean(key) / (MS_ENC_S + tmean(key))
            setv(f"ms.share.{m}", sh, 1)
            shares.append(100 - sh)
    if shares:
        setv("ms.share.enc.min", min(shares), 0)
        setv("ms.share.enc.max", max(shares), 0)
    if "hier" in rows:
        for b, tag in ((L.B1, ""), (B_LO, ".q25")):
            g = np.mean([x[f"ann_recall10@scan{b}"] for x in rows["cr_ep5"]]) - np.mean([x[f"ann_recall10@scan{b}"] for x in rows["hier"]])
            setv(f"ms.gain.ann{tag}", 100 * g, 2)


MS_ROWS_M = [(key, m) for key, m, g in MS_SUPP]


# ---------------------------------------------------------------- 维护（RQ4，3 个种子，墙钟时间，对照实验，检验）
STRAT = [("frozen", "frozen", GRAY_L, "F"), ("local_lloyd", "local Lloyd", ORANGE, "LL"), ("cert_lloyd", "certified Lloyd", PURPLE, "CL"),
         ("cert_hart", "certified Hartigan", BLUE, "CH")]
STRAT_STYLE = {"frozen": ("o", "-", False), "local_lloyd": ("D", "-", False), "cert_lloyd": ("v", "-", False), "cert_hart": ("s", "-", False)}
WL = ["uniform", "clustered", "drift"]
MAINT_SEEDS = [42, 1, 2]


def maint():
    res, ctrl = {}, {}
    for f in glob.glob(os.path.join(L.RES, "*", "maint", "*_s*.json")):
        m = re.match(r"(.+)_s(\d+)(_control)?\.json$", os.path.basename(f))
        if not m:
            continue
        r = json.load(open(f))
        if m.group(3):
            ctrl[(r["ds"], r["workload"])] = r
        else:
            res.setdefault((r["ds"], r["workload"]), {})[int(m.group(2))] = r
    if not res:
        missing("maint results")
        return
    keys = sorted(res, key=lambda k: (DS.index(k[0]) if k[0] in DS else 99, WL.index(k[1]) if k[1] in WL else 9))
    for k in keys:
        lack = [s for s in MAINT_SEEDS if s not in res[k]]
        if lack:
            missing(f"maint {k[0]}/{k[1]} seeds {lack}")
        if k not in ctrl:
            missing(f"maint {k[0]}/{k[1]}_s42_control.json")
    setv("maint.seeds", min(len(v) for v in res.values()))
    setv("maint.seeds.max", max(len(v) for v in res.values()))
    setv("stat.seeds.maint", min(len(v) for v in res.values()))

    def fin(r, s):
        return r["hist"][-1][s]["recall@scan0.01"]

    def cost_pct(r, s):
        full = r["hist"][-1]["full_pass_dims_per_point"]
        return float(np.mean([h[s + "_cost"]["dims_per_point"] for h in r["hist"][1:]]) / full * 100)

    def wall_pct(r, s):
        return 100 * r["maint_total_s"][s] / r["batches"] / r["rebuild"]["build_s"]

    def wall_total(r, s):
        return 100 * r["maint_total_s"][s] / r["rebuild"]["build_s"]

    def wall_per10k(r, s):
        return 100 * r["maint_total_s"][s] / r["n_insert"] * 10000 / r["rebuild"]["build_s"]

    def mean(rs, fn):
        v = [fn(r) for r in rs]
        return float(np.nanmean(v)) if v and not all(np.isnan(v)) else np.nan

    def rstr(v):
        return "--" if v is None or np.isnan(v) else f"{100 * v:.2f}"
    # 逐行（数据集 × 负载）的种子均值 + 对 CH 的检验（逐查询 pqb 数组，≥2 个种子）
    M, tests, fam_m = {}, {}, set()
    for k in keys:
        rs = list(res[k].values())
        row = dict(seeds=len(rs), rec={ab: mean(rs, lambda r: fin(r, s)) for s, _, _, ab in STRAT},
                   recR=mean(rs, lambda r: r.get("rebuild", {}).get("recall@scan0.01", np.nan)),
                   maxl={ab: mean(rs, lambda r: r["hist"][-1][s]["max_list"]) for s, _, _, ab in STRAT},
                   maxlR=mean(rs, lambda r: r.get("rebuild", {}).get("max_list", np.nan)),
                   work={ab: mean(rs, lambda r: cost_pct(r, s)) for s, _, _, ab in STRAT[1:]},
                   wall={ab: mean(rs, lambda r: wall_pct(r, s)) for s, _, _, ab in STRAT[1:]},
                   wall_total={ab: mean(rs, lambda r: wall_total(r, s)) for s, _, _, ab in STRAT[1:]},
                   wall_per10k={ab: mean(rs, lambda r: wall_per10k(r, s)) for s, _, _, ab in STRAT[1:]},
                   rebuild_s=mean(rs, lambda r: r["rebuild"]["build_s"]), ctrl={})
        # 对照：不插入、只对已建索引跑 CL/CH（<w>_s42_control.json）
        if k in ctrl:
            c = ctrl[k]
            r42 = res[k].get(42)
            built = (r42 or c)["hist"][0]["built"]["recall@scan0.01"]
            cb = c.get("hist", [{}])[0].get("built", {}).get("recall@scan0.01", built)
            if r42 and abs(cb - built) > 1e-6:
                print("WARNING: control run and maintenance run start from different built indexes on", k)
                setv("maint.control.samebase", False)
            else:
                V.setdefault("maint.control.samebase", "yes")
            for s, _, _, ab in STRAT[2:]:
                v = None
                for probe in (lambda: c["final"][s]["recall@scan0.01"], lambda: c[s]["recall@scan0.01"], lambda: c["hist"][-1][s]["recall@scan0.01"]):
                    try:
                        v = probe()
                        break
                    except (KeyError, TypeError, IndexError):
                        continue
                if v is None:
                    print("WARNING: unknown control file layout for", k)
                else:
                    row["ctrl"][ab] = 100 * (v - built)
                    setv(f"maint.control.{k[0]}.{k[1]}.{ab}.gain", row["ctrl"][ab], 2)
        M[k] = row
        # 检验：CH 减 F/LL/CL/R，1% 预算列，两层自助法，行内 Holm
        d = os.path.join(L.RES, k[0], "maint")
        arrs = {}
        for s, _, _, ab in STRAT:
            mats = [np.load(os.path.join(d, f"{k[1]}_s{sd}_{s}.pqb.npy")) for sd in sorted(res[k]) if os.path.exists(os.path.join(d, f"{k[1]}_s{sd}_{s}.pqb.npy"))]
            if len(mats) >= 2:
                arrs[ab] = np.vstack([mm[:, L.BUDGETS.index(L.B1)].astype(np.float64) for mm in mats])
        mats = [np.load(os.path.join(d, f"{k[1]}_s{sd}_rebuild.pqb.npy")) for sd in sorted(res[k]) if os.path.exists(os.path.join(d, f"{k[1]}_s{sd}_rebuild.pqb.npy"))]
        if len(mats) >= 2:
            arrs["R"] = np.vstack([mm[:, L.BUDGETS.index(L.B1)].astype(np.float64) for mm in mats])
        if "CH" in arrs and len(arrs) >= 2:
            others = [ab for ab in arrs if ab != "CH"]
            bts = [L.boot_diff(arrs["CH"], arrs[ab], n_boot=N_BOOT, delta=DELTA) for ab in others]
            fam_m.add(len(others))
            for ab, bt, pa in zip(others, bts, L.holm([bt["p"] for bt in bts])):
                tests[(k, ab)] = dict(diff=100 * bt["diff"], p_holm=float(pa), sig=bool(pa < 0.05), equiv=bool(bt["equiv"]), cls=classify(bt, pa))
        else:
            missing(f"maint {k[0]}/{k[1]} per-query pqb arrays (>=2 seeds)")
    if fam_m:
        setv("fam.maint", max(fam_m))
        setv("fam.maint.min", min(fam_m))
    SUMMARY["maint"] = {f"{k[0]}|{k[1]}": v for k, v in M.items()}
    SUMMARY["maint_tests"] = {f"{k[0]}|{k[1]}|{ab}": v for (k, ab), v in tests.items()}
    has_ctrl = any(M[k]["ctrl"] for k in keys)

    def mark(k, ab):
        t = tests.get((k, ab))
        if not t:
            return ""
        sup = ("\\blacktriangle" if t["diff"] < 0 else "\\triangledown") if t["sig"] else ""
        sup += "=" if t["equiv"] else ""
        return r"$^{" + sup + "}$" if sup else ""
    # 表：列 = 负载（按集合分组），行 = 策略，分块给出召回、最大列表、距离工作量、墙钟与无插入对照
    names = {"F": "Frozen (F)", "LL": "Local Lloyd (LL)", "CL": "Cert.\\ Lloyd (CL)", "CH": "Cert.\\ Hartigan (CH)", "R": "Rebuild (R)"}
    groups = []
    for k in keys:
        if groups and groups[-1][0] == k[0]:
            groups[-1][1] += 1
        else:
            groups.append([k[0], 1])
    ncol = 1 + len(keys)
    hdr = [" & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{L.DSNAME.get(ds, ds)}}}" for ds, n in groups) + r" \\"]
    cm, c0 = [], 2
    for ds, n in groups:
        cm.append(rf"\cmidrule(lr){{{c0}-{c0 + n - 1}}}")
        c0 += n
    hdr += ["".join(cm), "Strategy & " + " & ".join(k[1] for k in keys) + r" \\"]
    body = []

    def block(title, rows, first=False):
        if not first:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{title}}}}} \\")
        for lab, cells in rows:
            body.append(lab + " & " + " & ".join(cells) + r" \\")
    abs_all = [ab for _, _, _, ab in STRAT]
    block(r"Recall@10 (\%) at a 1\% scan budget",
          [(names[ab], [rstr(M[k]["rec"][ab]) + mark(k, ab) for k in keys]) for ab in abs_all] +
          [(names["R"], [rstr(M[k]["recR"]) + mark(k, "R") for k in keys])], first=True)
    block("Largest list (vectors)",
          [(names[ab], [f"{M[k]['maxl'][ab]:,.0f}" for k in keys]) for ab in abs_all] +
          [(names["R"], ["--" if np.isnan(M[k]["maxlR"]) else f"{M[k]['maxlR']:,.0f}" for k in keys])])
    block(r"Distance computations per batch (\% of an uncertified pass)",
          [(names[ab], [f"{M[k]['work'][ab]:.0f}" for k in keys]) for ab in abs_all[1:]])
    block(r"Wall-clock time per batch (\% of a rebuild)",
          [(names[ab], [f"{M[k]['wall'][ab]:.1f}" for k in keys]) for ab in abs_all[1:]])
    if has_ctrl:
        block("No-insertion gain (recall points, seed 42)",
              [(names[ab], [fmt(M[k]["ctrl"].get(ab), 2) if ab in M[k]["ctrl"] else "--" for k in keys]) for ab in ("CL", "CH")])
    nan_note = "; --: a single list exceeds the budget" if any(np.isnan(v) for r in M.values() for v in r["rec"].values()) else ""
    write_gen("tab_maint.tex", table(
        "table*", "Maintenance after the last insertion batch.", "tab:maint", "l" + "r" * len(keys), hdr, body, tabcolsep="5pt",
        note=((r"$^{\blacktriangle}$/$^{\triangledown}$: significantly higher/lower than CH; $^{=}$: equivalent to CH within $\pm$\V{stat.delta.pts} points"
               if tests else "") + nan_note + ".").lstrip("; ")))
    # 正文数字
    for _, _, _, ab in STRAT[1:]:
        for nm, fld in (("cost", "work"), ("wall", "wall"), ("wall.total", "wall_total"), ("wall.per10k", "wall_per10k")):
            vals = [M[k][fld][ab] for k in keys if not np.isnan(M[k][fld][ab])]
            if vals:
                setv(f"maint.{nm}.{ab}.min", min(vals), 0 if nm == "cost" else 1)
                setv(f"maint.{nm}.{ab}.max", max(vals), 0 if nm == "cost" else 1)
        vals = [M[k]["wall"][ab] for k in keys if k[1] == "uniform" and not np.isnan(M[k]["wall"][ab])]
        if vals:
            setv(f"maint.wall.{ab}.uniform.min", min(vals), 1)
            setv(f"maint.wall.{ab}.uniform.max", max(vals), 1)
    for k in keys:
        w = {ab: M[k]["wall"][ab] for _, _, _, ab in STRAT[1:] if not np.isnan(M[k]["wall"][ab])}
        if w:
            setv(f"maint.wall.cheapest.{k[0]}.{k[1]}", min(w, key=w.get))
            wc = {ab: v for ab, v in w.items() if ab in ("CL", "CH")}
            if wc:
                setv(f"maint.wall.cheapest.cert.{k[0]}.{k[1]}", min(wc, key=wc.get))
    ind = [M[k] for k in keys if k[1] in ("uniform", "clustered")]
    if ind:
        setv("maint.ind.spread.max", 100 * max(max(r["rec"].values()) - min(r["rec"].values()) for r in ind), 2)
        vsreb = [r["rec"]["CH"] - r["recR"] for r in ind if not np.isnan(r["recR"])]
        if vsreb:
            setv("maint.ind.chvsreb.min", 100 * min(vsreb), 2)
            setv("maint.ind.chvsreb.max", 100 * max(vsreb), 2)
        tr = [r["wall_total"]["CH"] for r in ind if not np.isnan(r["wall_total"]["CH"])]
        if tr:
            setv("maint.ind.time.ch.pct.max", max(tr), 0)
    unif = [M[k] for k in keys if k[1] == "uniform" and not np.isnan(M[k]["recR"])]
    if unif:
        setv("maint.unif.spread.max", 100 * max(max(r["rec"].values()) - min(r["rec"].values()) for r in unif), 1)
        setv("maint.unif.vsreb.max", 100 * max(max(abs(v - r["recR"]) for v in r["rec"].values()) for r in unif), 1)
    clus = [M[k] for k in keys if k[1] == "clustered"]
    if clus:
        g = [r["rec"]["CH"] - r["rec"]["F"] for r in clus]
        setv("maint.clus.chgain.min", 100 * min(g), 2)
        setv("maint.clus.chgain.max", 100 * max(g), 2)
        setv("maint.clus.chbest.count", sum(1 for r in clus if max(r["rec"], key=r["rec"].get) == "CH"))
        setv("maint.clus.n", len(clus))
    drift = [M[k] for k in keys if k[1] == "drift" and not np.isnan(M[k]["recR"])]
    if drift:
        gaps = [r["recR"] - r["rec"]["F"] for r in drift]
        rec = [(r["rec"]["CH"] - r["rec"]["F"]) / (r["recR"] - r["rec"]["F"]) for r in drift]
        setv("maint.drift.gap.min", 100 * min(gaps), 1)
        setv("maint.drift.gap.max", 100 * max(gaps), 1)
        setv("maint.drift.recov.min", 100 * min(rec), 0)
        setv("maint.drift.recov.max", 100 * max(rec), 0)
        setv("maint.drift.chbest.count", sum(1 for r in drift if max(r["rec"], key=r["rec"].get) == "CH"))
        setv("maint.drift.n", len(drift))
        mx = [r["maxl"]["CH"] / r["maxlR"] for r in drift]
        setv("maint.drift.maxratio.min", min(mx), 1)
        setv("maint.drift.maxratio.max", max(mx), 1)
    cd = [M[k] for k in keys if k[1] in ("clustered", "drift")]
    if cd:
        setv("maint.ch.best.count", sum(1 for r in cd if max(r["rec"], key=r["rec"].get) == "CH"))
        setv("maint.ch.best.n", len(cd))
        if V["maint.ch.best.count"] != V["maint.ch.best.n"]:
            print("WARNING: certified Hartigan is not the best maintenance strategy in every clustered/drift run - check the RQ4 text")
    if tests:
        for ab in ("F", "LL", "CL", "R"):
            cl = [t["cls"] for (k, a), t in tests.items() if a == ab]
            if cl:
                for c in "WELI":
                    setv(f"maint.wel.ch.{ab}.{c}", cl.count(c))
    shares = []
    for k in keys:
        r = M[k]
        lead = 100 * (r["rec"]["CH"] - r["rec"]["F"])
        if "CH" in r["ctrl"] and lead > 0:
            shares.append(100 * r["ctrl"]["CH"] / lead)
    if shares:
        setv("maint.control.share.min", min(shares), 0)
        setv("maint.control.share.max", max(shares), 0)
    rts = [M[k]["rebuild_s"] for k in keys if not np.isnan(M[k]["rebuild_s"])]
    if rts:
        setv("maint.rebuild.t.min", min(rts), 1)
        setv("maint.rebuild.t.max", max(rts), 1)
    for k in keys:
        kk = f"maint.{k[0]}.{k[1]}"
        r = M[k]
        for _, _, _, ab in STRAT:
            setv(f"{kk}.{ab}", "--" if np.isnan(r["rec"][ab]) else f"{100 * r['rec'][ab]:.1f}")
            setv(f"{kk}.{ab}.max", f"{r['maxl'][ab]:,.0f}")
            if ab != "F":
                setv(f"{kk}.{ab}.wall", r["wall"][ab], 1)
        if not np.isnan(r["recR"]):
            setv(f"{kk}.R", 100 * r["recR"], 1)
            setv(f"{kk}.R.max", f"{r['maxlR']:,.0f}")
            setv(f"{kk}.R.t", r["rebuild_s"], 1)
        rs = list(res[k].values())
        for s, _, _, ab in STRAT:
            best = mean(rs, lambda r_: max((h[s]["recall@scan0.01"] for h in r_["hist"][1:] if not np.isnan(h[s]["recall@scan0.01"])), default=np.nan))
            setv(f"{kk}.{ab}.best", 100 * best, 1)
    # 图（补充材料）：逐批 recall 的种子均值
    nrow = max(1, (len(keys) + 2) // 3)
    fig, axes = plt.subplots(nrow, 3, figsize=FS(2.1 * nrow), squeeze=False)
    rng = {}
    for ax, k in zip(axes.ravel(), keys):
        style(ax)
        rs = list(res[k].values())
        nb = min(len(r["hist"]) for r in rs)
        xs = [rs[0]["hist"][i]["n"] / 1e3 for i in range(1, nb)]
        vals = []
        for s, lab, col, _ in STRAT:
            ys = [100 * float(np.nanmean([r["hist"][i][s]["recall@scan0.01"] for r in rs])) for i in range(1, nb)]
            mk, ls_, hol = STRAT_STYLE[s]
            ax.plot(xs, ys, marker=mk, ms=3.2, ls=ls_, color=col, mfc="white" if hol else col, mec=col if hol else "white",
                    mew=0.8 if hol else 0.4, label=lab)
            vals += [y for y in ys if not np.isnan(y)]
        if not np.isnan(M[k]["recR"]):
            ax.plot([xs[-1]], [100 * M[k]["recR"]], marker="*", ms=9, color=INK, ls="none", label="rebuild (reference)")
            vals.append(100 * M[k]["recR"])
        rng[k] = (min(vals), max(vals))
        ax.set_title(f"{L.DSNAME.get(k[0], k[0])}, {k[1]}", loc="left")
        ax.set_xlabel("indexed vectors (thousands)")
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    span = {}
    for k, (lo_, hi_) in rng.items():
        span[k[1]] = max(span.get(k[1], 1.0), hi_ - lo_)
    for ax, k in zip(axes.ravel(), keys):
        lo_, hi_ = rng[k]
        sp = span[k[1]]
        pad = 0.12 * sp
        mid = (lo_ + hi_) / 2
        ax.set_ylim(mid - sp / 2 - pad, mid + sp / 2 + pad)
        step = next(s for s in (0.2, 0.5, 1.0, 2.0, 5.0) if (sp + 2 * pad) / s <= 6)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    for ax in axes[:, 0]:
        ax.set_ylabel("recall@10 at 1% (%)")
    hh, ll = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, hh, ll, 5)
    fig.savefig(os.path.join(FIG, "fig_maint.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- FAISS 迭代次数 × 样本量扫描
FS_PAT = re.compile(r"^faiss_it(\d+)_(m(\d+)|full)$")
FS_REFS = [("oursHfive", ""), ("oursHL", ".hl"), ("oursBL", ".bl")]   # (认证配置的宏名, fsweep.<ds>.eqrecall<后缀>.* 键的后缀)


def fsweep():
    pts_all, any_sweep = {}, False
    for ds in DS:
        n = None
        for en in R.get(ds, {}).values():
            for run in en.values():
                if "refine" in run:
                    n = run["refine"]["n"]
                    break
            if n:
                break
        c = c_of_ds(ds)
        pts = {}
        for (m, e), en in R.get(ds, {}).items():
            mm = FS_PAT.match(m)
            if mm:
                any_sweep = True
                it, samp = int(mm.group(1)), (None if mm.group(2) == "full" else int(mm.group(3)))
            elif m == "faiss_ivf":
                it, samp = 10, 256
            elif m == "faiss":
                it, samp = 25, None
            elif m == "faiss_sampled" and ds == "deep10m":
                it, samp = 25, 256
            else:
                continue
            eff = n if (samp is None or n is None or samp * c >= n) else samp * c   # 有效样本量（m·c ≥ n 时等于全量）
            s_ = stat(ds, m, e)
            if not s_ or not isnum(s_["r"]):
                continue
            key = (it, eff)
            lab = f"{it} it., " + ("full" if eff == n else f"{samp} pts/centroid")
            if key in pts and not (samp is None):     # 相同 (iter, 有效样本) 的配置只保留一次，优先 *_full
                continue
            pts[key] = dict(t=s_["t"], r=s_["r"], t_sd=s_["t_sd"], r_sd=s_["r_sd"], n=s_["k"], label=lab, tag=m,
                            it=it, pts="full" if eff == n else str(samp))
        pts_all[ds] = pts
    if not any_sweep:
        missing("faiss_it{5,10,25}_m{39,64,128}/full sweep (no collection)")
    SUMMARY["fsweep"] = {ds: {f"{k[0]}|{k[1]}": v for k, v in p.items()} for ds, p in pts_all.items()}
    speeds, notreached, below1 = {}, [], []
    fs_best = {}   # (ds, 宏名) -> 最便宜的达标配置（None: 扫描里没有配置达到该召回）
    for ds, pts in pts_all.items():
        if not pts:
            continue
        setv(f"fsweep.seeds.{ds}", min(v["n"] for v in pts.values()))
        setv(f"fsweep.{ds}.configs", len(pts))
        for m, key in FS_REFS:
            s_ = stat(ds, *BY_MACRO[m])
            if not s_:
                continue
            ok = [v for v in pts.values() if v["r"] >= s_["r"]]
            if ok:
                best = min(ok, key=lambda v: v["t"])
                sp = best["t"] / s_["t"]
                setv(f"fsweep.{ds}.eqrecall{key}.speed", sp, 1)
                setv(f"fsweep.{ds}.eqrecall{key}.cfg", best["label"])
                fs_best[(ds, m)] = best
                if m == "oursHfive":
                    speeds[ds] = sp
                    if sp < 1:
                        below1.append(L.DSNAME[ds])
            else:
                setv(f"fsweep.{ds}.eqrecall{key}.speed", "--")
                setv(f"fsweep.{ds}.eqrecall{key}.cfg", "not reached")
                fs_best[(ds, m)] = None
                if m == "oursHfive":
                    notreached.append(L.DSNAME[ds])
    if speeds:
        setv("fsweep.eqrecall.speed.min", min(speeds.values()), 1)
        setv("fsweep.eqrecall.speed.max", max(speeds.values()), 1)
        setv("fsweep.reached.count", len(speeds))
    setv("fsweep.notreached.list", ", ".join(notreached) or "none")
    setv("fsweep.below1.list", ", ".join(below1) or "none")
    # 表：每个数据集一行；对每个认证配置给出最便宜的达标 FAISS 配置（迭代数、每个质心的样本点数）与时间比（= fsweep.<ds>.eqrecall*.speed）
    rows_fs = []
    for ds in DS:
        if not any((ds, m) in fs_best for m, _ in FS_REFS):
            continue
        cells = [L.DSNAME[ds], V[f"fsweep.seeds.{ds}"]]
        for m, key in FS_REFS:
            b = fs_best.get((ds, m), False)
            if b is False:
                cells.append(r"\multicolumn{3}{c}{--}")
            elif b is None:
                cells.append(r"\multicolumn{3}{c}{not reached}")
            else:
                cells += [str(b["it"]), b["pts"], V[f"fsweep.{ds}.eqrecall{key}.speed"]]
        rows_fs.append(" & ".join(cells) + r" \\")
    if rows_fs:
        write_gen("tab_fsweep.tex", table(
            "table*", r"Cheapest FAISS configuration at equal recall.", "tab:fsweep", "lr" + "rrr" * len(FS_REFS),
            [" & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{tex(m)}}}" for m, _ in FS_REFS) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(FS_REFS))),
             "Collection & Seeds & " + " & ".join(["iter. & pts/centroid & speedup"] * len(FS_REFS)) + r" \\"], rows_fs, tabcolsep="3pt",
            note="iter.: Lloyd iterations; pts/centroid: sampled points per centroid (full: all vectors); "
                 "speedup: build time of the FAISS configuration divided by that of the certified configuration."))
    if not any(pts_all.values()):
        return
    fig, axes = plt.subplots(2, 4, figsize=FS(4.2))
    handles = {}
    for ax, ds in zip(axes.ravel(), DS):
        style(ax)
        pts = pts_all.get(ds, {})
        if not pts:
            ax.set_visible(False)
            continue
        xs = np.array([v["t"] for v in pts.values()])
        ys = np.array([100 * v["r"] for v in pts.values()])
        h = ax.plot(xs, ys, "o", color=TEAL, alpha=0.6, ms=4.2, mec="white", mew=0.5, zorder=2)
        handles.setdefault("FAISS configurations", h[0])
        o = np.argsort(xs)
        env_x, env_y, best = [], [], -1
        for i in o:
            if ys[i] > best:
                env_x.append(xs[i])
                env_y.append(ys[i])
                best = ys[i]
        if len(env_x) > 1:
            h = ax.step(env_x, env_y, where="post", color=TEAL, lw=0.9, ls=(0, (3, 2)), zorder=1)
            handles.setdefault("FAISS upper envelope", h[0])
        for m in ("oursHL", "oursHone", "oursHfive", "oursBL", "oursBfive"):
            s_ = stat(ds, *BY_MACRO[m])
            if s_ and isnum(s_["r"]):
                col, ls = FAM_STYLE[FAMILY_OF[m]]
                hollow = False                              # B 族用自己的颜色（紫），不再靠空心区分
                h = ax.plot(s_["t"], 100 * s_["r"], MARKER[m], color=col, ms=6, mec=col if hollow else "white", mew=1.0 if hollow else 0.6,
                            mfc="white" if hollow else col, zorder=3)
                handles.setdefault(PRINTED[m], h[0])
        ax.set_xscale("log")
        lo_, hi_ = ax.get_xlim()
        if hi_ / lo_ < 10:   # 不足一个数量级时对数刻度会把 1..9 全标出来：扩到一个数量级
            c_ = np.sqrt(lo_ * hi_)
            ax.set_xlim(c_ / np.sqrt(10), c_ * np.sqrt(10))
        ax.set_title(L.DSNAME[ds], loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 3.0)))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    for ax in axes[:, 0]:
        ax.set_ylabel("recall@10 at 1% scan (%)")
    for ax in axes[-1]:
        ax.set_xlabel("build time (s)")
    top_legend(fig, list(handles.values()), list(handles), 4)
    fig.savefig(os.path.join(FIG, "fig_fsweep.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- 计时离群规则（tools/timing_outliers.py 的规则，进程内重算；只看主表的 18 个配置）
RET_TAGS = {("hier", None): "hier", ("cr", 5): "certrefine", ("cr_labels", 5): "certrefine", ("hslc", 5): "certrefine", ("hslc_labels", 5): "certrefine",
            ("hsl", 5): "certrefine", ("sl", 5): "certrefine", ("hb", 5): "certrefine", ("hsl_cpp", None): "hslskm", ("hor", None): "hor",
            ("hor1", None): "hor", ("hor2", None): "hor", ("warm1", None): "ws", ("warm2", None): "ws", ("warm3", None): "ws",
            ("skm_default", None): "skmdef", ("faiss_ivf", None): "faiss", ("faiss", None): "faiss", ("lloyd", None): "lloyd"}


def timing_flags(tol=0.15, min_seeds=5):
    flag = []
    for ds in DS:
        for (m, e), grp in RET_TAGS.items():
            en = entry(ds, m, e)
            if not en:
                continue
            t = {s: v["time_train"] for s, v in en.items() if not np.isnan(v["time_train"])}
            if len(t) < min_seeds:
                continue
            for s, x in t.items():
                ref = float(np.median([y for s2, y in t.items() if s2 != s]))
                r = x / ref
                if r > 1 + tol or r < 1 / (1 + tol):
                    flag.append((ds, m, e, s, x, ref, r, grp))
    lines = [f"# timing-outlier rule: |ratio to the median of the other seeds| beyond +-{100 * tol:.0f}%, main-table configurations with >= {min_seeds} seeds, complete runs only",
             "# collection:method:seed   time  median(other seeds)  ratio"]
    lines += [f"{ds}:{m}{'' if e is None else f'_ep{e}'}:{s}   time={x:.2f}s  median={ref:.2f}s  ratio={r:.2f}" for ds, m, e, s, x, ref, r, grp in flag]
    lines.append(f"# {len(flag)} run(s) flagged")
    write_gen("timing_deviations.txt", "\n".join(lines) + "\n")
    SUMMARY["timing_flagged"] = [dict(ds=ds, method=m, epoch=e, seed=s, time=x, median=ref, ratio=r) for ds, m, e, s, x, ref, r, grp in flag]
    setv("ret.flagged.count", len(flag))
    for g in ("hor", "skmdef", "lloyd", "hier", "certrefine", "hslskm", "ws", "faiss"):
        setv(f"ret.flagged.{g}", sum(1 for f in flag if f[7] == g))
    setv("ret.flagged.bcdkm", 0)   # bcdkm 不在主表配置里，规则不覆盖
    if flag:
        setv("ret.flagged.ratio.min", min(f[6] for f in flag), 2)
        setv("ret.flagged.ratio.max", max(f[6] for f in flag), 2)
        setv("ret.flagged.list", "; ".join(f"{L.DSNAME[f[0]]} {f[1]} seed {f[3]} ({f[6]:.2f})" for f in flag))
    setv("ret.tol", 100 * tol, 0)
    setv("ret.minseeds", min_seeds)


# ---------------------------------------------------------------- SKM-default 的查询泄漏修正：与 _superseded 的运行比较
def leak_numbers():
    dts, drs, n = {}, [], 0
    for ds in DS + ["msmarco"]:
        d = os.path.join(L.RES, ds)
        sd = os.path.join(d, "_superseded")
        if not os.path.isdir(sd):
            continue
        ratios = []
        for s in L.SEEDS:
            la, lb = os.path.join(d, f"skm_default_s{s}.log"), os.path.join(sd, f"skm_default_s{s}.log")
            if not (os.path.exists(la) and os.path.exists(lb)):
                continue
            ta, tb = L.train_s(la), L.train_s(lb)
            if ds == "msmarco":
                ja, jb = os.path.join(d, f"skm_default_s{s}.ir.json"), os.path.join(sd, f"skm_default_s{s}.ir.json")
                key = "ann_recall10@scan0.01"
            else:
                ja, jb = os.path.join(d, f"skm_default_s{s}.scan.json"), os.path.join(sd, f"skm_default_s{s}.scan.json")
                key = "recall@scan0.01"
            if not (os.path.exists(ja) and os.path.exists(jb)):
                continue
            ra, rb = json.load(open(ja))[key], json.load(open(jb))[key]
            ratios.append(ta / tb)
            drs.append(100 * (ra - rb))
            n += 1
        if ratios:
            dts[ds] = float(np.mean(ratios))
            setv(f"leak.skmdef.{ds}.dt", dts[ds], 2)
    if n:
        setv("leak.skmdef.n", n)
        setv("leak.skmdef.r.min", min(drs), 2)
        setv("leak.skmdef.r.max", max(drs), 2)
        setv("leak.skmdef.dt.min", min(dts.values()), 2)
        setv("leak.skmdef.dt.max", max(dts.values()), 2)
    else:
        missing("_superseded skm_default runs (leak numbers)")
    SUMMARY["leak"] = dict(dt=dts, dr=drs, n=n)


# ---------------------------------------------------------------- 最终分配的三种做法（若有 crf/crf_labels/bf_labels 运行）
FIN_VARS = [("crf", "oursHfive", "cr"), ("crf_labels", "oursHL", "cr_labels"), ("bf_labels", "oursBL", "hslc_labels")]


def final_assignment():
    rows, have = [], False
    ll, sl, hl_sp = [], [], []
    for ds in DS + ["msmarco"]:
        d = os.path.join(L.RES, ds)
        if not os.path.isdir(d):
            continue
        fa = L.assign_s(d)
        cells = [L.DSNAME.get(ds, ds)]
        for tag, m, base in FIN_VARS:
            js = sorted(glob.glob(os.path.join(d, f"{tag}_s*.refine.json")))
            if not js:
                cells += ["--"] * 3
                continue
            have = True
            ex_t, ex_r, li_t, li_r, sn_t, sn_r = [], [], [], [], [], []
            for p in js:
                seed = int(re.search(r"_s(\d+)\.refine", p).group(1))
                r = json.load(open(p))
                tot = r["hist"][5]["total_s"]
                key = "ann_recall10@scan0.01" if ds == "msmarco" else "recall@scan0.01"
                def rec(nm):
                    for ext in (".scan.json", ".ir.json"):
                        q = os.path.join(d, nm + ext)
                        if os.path.exists(q):
                            return json.load(open(q)).get(key, np.nan)
                    return np.nan
                li_t.append(tot)
                li_r.append(rec(f"{tag}_s{seed}_ep5"))
                sn_t.append(tot + r["timing"].get("final_snap_s", np.nan))
                sn_r.append(rec(f"{tag}_s{seed}_ep5snap"))
                ex_t.append(tot + fa)
                ex_r.append(rec(f"{base}_s{seed}_ep5"))
            vals = {}
            for op, ts, rs in (("exact", ex_t, ex_r), ("lists", li_t, li_r), ("snap", sn_t, sn_r)):
                t_, r_ = float(np.nanmean(ts)), 100 * float(np.nanmean(rs))
                vals[op] = (t_, r_)
                setv(f"fin.{ds}.{m}.{op}.t", t_, 1 if t_ < 100 else 0)
                setv(f"fin.{ds}.{m}.{op}.r", r_, 2)
                cells.append(f"{tstr(t_)} / {fmt(r_, 2)}")
            if not np.isnan(vals["lists"][1]) and not np.isnan(vals["exact"][1]):
                ll.append(vals["exact"][1] - vals["lists"][1])
            if not np.isnan(vals["snap"][1]) and not np.isnan(vals["exact"][1]):
                sl.append(vals["exact"][1] - vals["snap"][1])
            if m == "oursHL":
                fi = stat(ds, "faiss_ivf", None)
                if fi and not np.isnan(vals["lists"][0]):
                    hl_sp.append(fi["t"] / vals["lists"][0])
                if ds == "deep10m":
                    setv("fin.deep.hl.lists.t", vals["lists"][0], 0)
            if m == "oursBL" and ds == "msmarco":
                setv("fin.ms.bl.lists.t", vals["lists"][0], 0)
        rows.append(cells)
    if not have:
        missing("crf/crf_labels/bf_labels runs (final-assignment table)")
        return
    if ll:
        setv("fin.lists.loss.min", min(ll), 2)
        setv("fin.lists.loss.max", max(ll), 2)
    if sl:
        setv("fin.snap.loss.min", min(sl), 2)
        setv("fin.snap.loss.max", max(sl), 2)
    if hl_sp:
        setv("fin.hl.lists.speed.fivf.min", min(hl_sp), 1)
        setv("fin.hl.lists.speed.fivf.max", max(hl_sp), 1)
    hdr = [r" & \multicolumn{9}{c}{build time (s) / recall@10 (\%)} \\", r"\cmidrule(lr){2-10}",
           r" & \multicolumn{3}{c}{\oursHfive} & \multicolumn{3}{c}{\oursHL} & \multicolumn{3}{c}{\oursBL} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
           r"Collection & exact & lists & snap & exact & lists & snap & exact & lists & snap \\"]
    write_gen("tab_final.tex", table(
        "table*", "Final-assignment variants.", "tab:final", "l" + "c" * 9, hdr, [" & ".join(r) + r" \\" for r in rows], tabcolsep="3pt"))


# ---------------------------------------------------------------- 讨论用数字：Voronoi 一致率、C++ 移植的耗时比
def discussion_numbers():
    vc, vs = {}, {}
    for ds in DS:
        for tag, dst in (("vor_cr", vc), ("vor_sl", vs)):
            r = refine_json(ds, tag, 42)
            if r and "final_voronoi_agree" in r.get("diag", {}):
                dst[ds] = 100 * r["diag"]["final_voronoi_agree"]
    SUMMARY["voronoi"] = dict(hartigan=vc, lloyd_seq=vs)
    if vc:
        setv("disc.vor.cr.min", min(vc.values()), 1)
        setv("disc.vor.cr.max", max(vc.values()), 1)
    if vs:
        setv("disc.vor.sl.min", min(vs.values()), 1)
        setv("disc.vor.sl.max", max(vs.values()), 1)
    both = [ds for ds in vc if ds in vs]
    if both:
        setv("disc.vor.diff.max", max(abs(vc[ds] - vs[ds]) for ds in both), 1)

    def mean_of(en, key):
        v = [en[s]["scan"][key] for s in L.SEEDS if s in en and en[s]["scan"]]
        return float(np.mean(v)) if v else np.nan
    obj = {}
    for ds in DS:
        c5, fa, lz = entry(ds, "cr", 5), entry(ds, "faiss", None), entry(ds, "lloyd", None)
        if not (c5 and fa and lz):
            continue
        wc, wf, wl = mean_of(c5, "wcss"), mean_of(fa, "wcss"), mean_of(lz, "wcss")
        rc, rf = mean_of(c5, "recall@scan0.01"), mean_of(fa, "recall@scan0.01")
        obj[ds] = dict(full_lower_pct=100 * (1 - min(wf, wl) / wc), faiss_lower_pct=100 * (1 - wf / wc),
                       recall_cr5_minus_faiss=100 * (rc - rf), flip=bool((wf < wc) != (rf > rc)))
    SUMMARY["objective_vs_recall"] = obj
    if obj:
        setv("disc.wcss.fulllower.count", sum(1 for v in obj.values() if v["full_lower_pct"] > 0))
        setv("disc.wcss.fulllower.max", max(v["full_lower_pct"] for v in obj.values()), 1)
        setv("disc.flip.count", sum(1 for v in obj.values() if v["flip"]))
        setv("disc.obj.n", len(obj))
        if "gist" in obj:
            setv("disc.gist.wcss.pct", obj["gist"]["faiss_lower_pct"], 1)
            setv("disc.gist.recall.gap", obj["gist"]["recall_cr5_minus_faiss"], 1)
    ratio = {}
    for ds in DS:
        ja, jb = refine_json(ds, "xcr", 42), refine_json(ds, "cr", 42)
        if ja and jb:
            # 只比较证书精修的各轮（本文实现的部分）；起步分配在原型里调用 FAISS（C++），在移植版里是 OpenBLAS GEMM，不属于原型与否的差别
            ta = sum(h["epoch_s"] for h in ja["hist"][1:])
            tb = sum(h["epoch_s"] for h in jb["hist"][1:])
            tta = ta + ja["timing"]["setup_s"] + ja["timing"]["graph0_s"] + ja["timing"]["init_assign_s"]
            ttb = tb + jb["timing"]["setup_s"] + jb["timing"]["graph0_s"] + jb["timing"]["init_assign_s"]
            jn = refine_json(ds, "xcr_nocert", 42)
            ratio[ds] = dict(cpp_over_numba=ta / tb, cpp_over_numba_total=tta / ttb,
                             cpp_ident=all(x["labels_hash"] == y["labels_hash"] for x, y in zip(ja["hist"], jn["hist"])) if jn else None)
    SUMMARY["cpp_port"] = ratio
    pc = os.path.join(L.RES, "coloring.json")
    if os.path.exists(pc):
        col = json.load(open(pc))
        SUMMARY["coloring"] = col
        setv("color.min", min(v["clusters_per_color_mean"] for v in col.values()), 1)
        setv("color.max", max(v["clusters_per_color_mean"] for v in col.values()), 0)
        setv("color.ncol.min", min(v["colors"] for v in col.values()))
        setv("color.ncol.max", max(v["colors"] for v in col.values()))
    if ratio:
        setv("cpp.ratio.min", min(v["cpp_over_numba"] for v in ratio.values()), 2)
        setv("cpp.ratio.max", max(v["cpp_over_numba"] for v in ratio.values()), 2)
        idn = [v["cpp_ident"] for v in ratio.values() if v["cpp_ident"] is not None]
        if idn:
            setv("cpp.ident.count", sum(idn))
            setv("cpp.ident.total", len(idn))
            setv("cpp.ident", all(idn))
    rd, sd = [], []
    for ds in DS:
        a, b = entry(ds, "xcr", 5), entry(ds, "cr", 5)
        if a and b and 42 in a and 42 in b and a[42]["scan"] and b[42]["scan"]:
            rd.append(100 * (a[42]["scan"]["recall@scan0.01"] - b[42]["scan"]["recall@scan0.01"]))
            ha, hb = a[42]["hist"], b[42]["hist"]
            if "skipped_pct" in ha and "skipped_pct" in hb:
                sd.append(abs(ha["skipped_pct"] - hb["skipped_pct"]))
    if rd:
        setv("cpp.recall.diff.min", min(rd), 2)
        setv("cpp.recall.diff.max", max(rd), 2)
    rng_ = []
    for ds in DS:
        en = entry(ds, "cr", 5)
        if en:
            v = [en[s]["scan"]["recall@scan0.01"] for s in L.SEEDS if s in en and en[s]["scan"]]
            if len(v) >= 2:
                rng_.append(100 * (max(v) - min(v)))
    if rng_:
        setv("cr5.seedrange.max", max(rng_), 2)
    if sd:
        setv("cpp.skip.diff.max", max(sd), 1)


# ---------------------------------------------------------------- 消融（全部种子）：判据 Hartigan/Lloyd × 更新 顺序/批量，第 1 轮与第 5 轮
ABL = [("cr", "SH"), ("sl", "SL"), ("hb", "BH"), ("hslc", "BL")]


def ablation_table():
    rows, per = [], {}
    for ds in DS:
        ens = {(t, e): entry(ds, t, e) for t, _ in ABL for e in (1, 5)}
        if not all(ens.values()):
            missing(f"{ds} ablation runs (cr/sl/hb/hslc)")
            continue
        sd_ = [s for s in L.SEEDS if all(s in en and en[s]["scan"] for en in ens.values())]
        if len(sd_) < 3:
            missing(f"{ds} ablation seeds (<3)")
            continue
        mean = {k: 100 * np.mean([en[s]["scan"]["recall@scan0.01"] for s in sd_]) for k, en in ens.items()}
        keys, ps, dms = [], [], []
        for e in (1, 5):
            ref = L.per_seed_pq(ens[("cr", e)], seeds=sd_)
            for t, _ in ABL[1:]:
                bt = L.boot_diff(L.per_seed_pq(ens[(t, e)], seeds=sd_), ref, n_boot=N_BOOT, delta=DELTA)
                keys.append((t, e))
                ps.append(bt["p"])
                dms.append(bt["diff"])
        adj = L.holm(ps)
        setv("fam.abl", len(ps))
        mark = {k: (("+" if dm > 0 else "-") if pa < 0.05 else "") for k, pa, dm in zip(keys, adj, dms)}
        row = [L.DSNAME[ds]]
        for e in (1, 5):
            best = max(mean[(t, e)] for t, _ in ABL)
            for t, _ in ABL:
                txt = f"{mean[(t, e)]:.2f}"
                if abs(mean[(t, e)] - best) < 5e-5:
                    txt = r"\textbf{" + txt + "}"
                mk = mark.get((t, e), "")
                if mk:
                    txt += r"$^{" + ("\\blacktriangle" if mk == "+" else "\\triangledown") + "}$"
                row.append(txt)
        rows.append(row)
        per[ds] = dict(mean={f"{t}_{e}": v for (t, e), v in mean.items()}, mark={f"{t}_{e}": m for (t, e), m in mark.items()}, seeds=len(sd_))
    if not rows:
        return
    write_gen("tab_ablation.tex", table(
        "table", "Effect of the criterion and the update rule.", "tab:ablation", "lrrrrrrrr",
        [r" & \multicolumn{4}{c}{recall@10 (\%) after pass 1} & \multicolumn{4}{c}{recall@10 (\%) after pass 5} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Collection & SH & SL & BH & BL & SH & SL & BH & BL \\"], [" & ".join(r) + r" \\" for r in rows],
        note=r"S/B: sequential/batch update; H/L: Hartigan/Lloyd criterion; $^{\blacktriangle}$/$^{\triangledown}$: significantly "
             r"higher/lower than SH; bold: best."))
    SUMMARY["ablation3"] = per
    n = len(per)
    m1 = {ds: v["mean"] for ds, v in per.items()}
    seqwins = sum(1 for m in m1.values() if max(m["cr_1"], m["sl_1"]) > max(m["hb_1"], m["hslc_1"]))
    shbest1 = sum(1 for m in m1.values() if m["cr_1"] >= max(m[f"{t}_1"] for t, _ in ABL) - 5e-5)
    spread5 = max(max(m[f"{t}_5"] for t, _ in ABL) - min(m[f"{t}_5"] for t, _ in ABL) for m in m1.values())
    sig5 = sum(1 for v in per.values() for k, mk in v["mark"].items() if k.endswith("_5") and mk)
    sig5_up = sum(1 for v in per.values() for k, mk in v["mark"].items() if k.endswith("_5") and mk == "+")
    sig1_dn = sum(1 for v in per.values() for k, mk in v["mark"].items() if k.endswith("_1") and mk == "-")
    sig1_up = sum(1 for v in per.values() for k, mk in v["mark"].items() if k.endswith("_1") and mk == "+")
    seq_gain1 = [m["cr_1"] - m["hb_1"] for m in m1.values()] + [m["sl_1"] - m["hslc_1"] for m in m1.values()]
    setv("abl.n", n)
    setv("abl.seeds", min(v["seeds"] for v in per.values()))
    setv("abl.p1.seqwins.count", seqwins)
    setv("abl.p1.shbest.count", shbest1)
    setv("abl.p1.sig.down", sig1_dn)
    setv("abl.p1.sig.up", sig1_up)
    setv("abl.p1.seqgain.min", min(seq_gain1), 2)
    setv("abl.p1.seqgain.max", max(seq_gain1), 2)
    setv("abl.p5.spread.max", spread5, 2)
    setv("abl.p5.sig.count", sig5)
    setv("abl.p5.sig.up", sig5_up)
    setv("abl.p5.n", 3 * n)
    setv("abl.p1.n", 3 * n)
    skip5, mvratio = {t: [] for t, _ in ABL}, []
    for ds in per:
        h = {}
        for t, _ in ABL:
            r = refine_json(ds, t, 42)
            if r:
                h[t] = r["hist"]
                skip5[t].append(h[t][5]["skipped_pct"])
        if "cr" in h and "sl" in h and h["sl"][5]["moves"]:
            mvratio.append(h["cr"][5]["moves"] / h["sl"][5]["moves"])
    for t, lab in ABL:
        if skip5[t]:
            setv(f"abl.skip5.{lab}.min", min(skip5[t]), 0)
            setv(f"abl.skip5.{lab}.max", max(skip5[t]), 0)
    if mvratio:
        setv("abl.mv5.ratio.min", min(mvratio), 1)
        setv("abl.mv5.ratio.max", max(mvratio), 1)
    SUMMARY["ablation_skip5"] = skip5
    SUMMARY["ablation_moves5_h_over_l"] = mvratio


def criterion_numbers():
    rows = {}
    for ds in DS:
        a, b = entry(ds, "cr", 5), entry(ds, "sl", 5)
        if not a or not b:
            missing(f"{ds} sl runs (criterion)")
            continue
        seeds = [s for s in L.SEEDS if s in a and s in b and a[s]["scan"] and b[s]["scan"]]
        if len(seeds) < 3:
            continue
        bt = L.boot_diff(L.per_seed_pq(b, seeds=seeds), L.per_seed_pq(a, seeds=seeds), n_boot=N_BOOT, delta=DELTA)
        dm, p = bt["diff"], bt["p"]
        wa = np.mean([a[s]["scan"]["wcss"] for s in seeds])
        wb = np.mean([b[s]["scan"]["wcss"] for s in seeds])
        h1 = np.mean([a[s]["hist"]["total_s"] for s in seeds])
        h2 = np.mean([b[s]["hist"]["total_s"] for s in seeds])
        e1a, e1b = entry(ds, "cr", 1), entry(ds, "sl", 1)
        p1 = (np.mean([e1a[s]["scan"]["recall@scan0.01"] for s in seeds if s in e1a and e1a[s]["scan"]]) -
              np.mean([e1b[s]["scan"]["recall@scan0.01"] for s in seeds if s in e1b and e1b[s]["scan"]])) if e1a and e1b else np.nan
        rows[ds] = dict(sl_minus_cr_pts=100 * dm, p=p, voronoi_wcss_sl_over_cr=wb / wa, time_cr=h1, time_sl=h2, pass1_cr_minus_sl_pts=100 * p1,
                        equiv=bt["equiv"], seeds=len(seeds))
    if not rows:
        return
    adj = L.holm([v["p"] for v in rows.values()])
    for (ds, v), pa in zip(rows.items(), adj):
        v["p_holm"] = float(pa)
        v["cls"] = classify(dict(diff=v["sl_minus_cr_pts"], equiv=v["equiv"]), pa)
    SUMMARY["criterion"] = rows
    setv("fam.crit", len(rows))
    setv("crit.sl.better.count", sum(1 for v in rows.values() if v["cls"] == "W"))
    setv("crit.cr.better.count", sum(1 for v in rows.values() if v["cls"] == "L"))
    setv("crit.equiv.count", sum(1 for v in rows.values() if v["cls"] == "E"))
    setv("crit.n", len(rows))
    setv("crit.seeds", min(v["seeds"] for v in rows.values()))
    setv("crit.diff.min", min(v["sl_minus_cr_pts"] for v in rows.values()), 2)
    setv("crit.diff.max", max(v["sl_minus_cr_pts"] for v in rows.values()), 2)
    setv("crit.wcss.sllower.count", sum(1 for v in rows.values() if v["voronoi_wcss_sl_over_cr"] < 1))
    setv("crit.wcss.crlower.count", sum(1 for v in rows.values() if v["voronoi_wcss_sl_over_cr"] > 1))
    setv("crit.wcss.pct.min", 100 * (min(v["voronoi_wcss_sl_over_cr"] for v in rows.values()) - 1), 1)
    setv("crit.wcss.pct.max", 100 * (max(v["voronoi_wcss_sl_over_cr"] for v in rows.values()) - 1), 1)
    setv("crit.time.sl.over.cr.min", min(v["time_sl"] / v["time_cr"] for v in rows.values()), 2)
    setv("crit.time.sl.over.cr.max", max(v["time_sl"] / v["time_cr"] for v in rows.values()), 2)
    sav = [100 * (1 - v["time_sl"] / v["time_cr"]) for v in rows.values()]
    setv("crit.time.slsaving.min", min(sav), 0)
    setv("crit.time.slsaving.max", max(sav), 0)
    p1 = [v["pass1_cr_minus_sl_pts"] for v in rows.values() if not np.isnan(v["pass1_cr_minus_sl_pts"])]
    if p1:
        setv("crit.pass1.min", min(p1), 2)
        setv("crit.pass1.max", max(p1), 2)
        setv("crit.pass1.crbetter.count", sum(1 for x in p1 if x > 0))


# ---------------------------------------------------------------- 系统：线程扩展、串行提交占比、批大小 B（seed 42）
def systems():
    thr = [1, 2, 5, 10, 20]
    rows_t, rows_b = [], []
    for ds in ("sift", "gist"):
        rr = {}
        for t in thr:
            r = refine_json(ds, f"cr_thr{t}", 42)
            if r:
                h = r["hist"][1:]
                if all("decide_s" in x for x in h):
                    rr[t] = dict(passes=sum(x["epoch_s"] for x in h), decide=sum(x["decide_s"] for x in h),
                                 commit=sum(x["commit_s"] for x in h), prep=sum(x["prep_s"] for x in h))
        if 1 in rr and len(rr) >= 3:
            for t in thr:
                if t in rr:
                    r = rr[t]
                    rows_t.append([L.DSNAME[ds], str(t), fmt(r["passes"], 2), f"{rr[1]['passes'] / r['passes']:.1f}",
                                   f"{100 * r['commit'] / r['passes']:.0f}", f"{100 * r['prep'] / r['passes']:.0f}"])
            SUMMARY.setdefault("threads", {})[ds] = rr
            best = max(t for t in rr)
            setv(f"sys.{ds}.speedup{best}", rr[1]["passes"] / rr[best]["passes"], 1)
            setv(f"sys.{ds}.commitshare{best}", 100 * rr[best]["commit"] / rr[best]["passes"], 0)
            setv(f"sys.{ds}.commitshare1", 100 * rr[1]["commit"] / rr[1]["passes"], 0)
        else:
            missing(f"{ds} cr_thr* runs (thread scaling)")
    for ds in ("sift", "glove200"):
        for T in (256, 1024, 4096, 16384):
            tag = "cr" if T == 4096 else f"cr_B{T}"
            en1, en5 = entry(ds, tag, 1), entry(ds, tag, 5)
            if not en1 or not en5 or 42 not in en1 or 42 not in en5 or not en1[42]["scan"] or not en5[42]["scan"]:
                missing(f"{ds} {tag}_s42 (batch-size sweep)")
                continue
            h = en5[42]["refine"]["hist"][1:]
            rej = 100 * sum(x["rejected"] for x in h) / max(1, sum(x["moves"] + x["rejected"] for x in h))
            rows_b.append([L.DSNAME[ds], str(T), f"{100 * en1[42]['scan']['recall@scan0.01']:.2f}",
                           f"{100 * en5[42]['scan']['recall@scan0.01']:.2f}", f"{rej:.2f}", fmt(sum(x["epoch_s"] for x in h), 2)])
    SUMMARY["batch_sweep"] = rows_b
    if not rows_t and not rows_b:
        return
    def sub(title, colspec, header, rows):
        return "\n".join([r"\begin{minipage}[t]{0.485\linewidth}", r"\centering", rf"\textit{{{title}}}\par\smallskip",
                          rf"\fittable{{{colspec}}}{{", r"\toprule"] + header + [r"\midrule"] +
                         [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule", r"}", r"\end{minipage}"])
    left = sub("(a) Threads (seed 42)", "lrrrrr",
               [r" & & & & \multicolumn{2}{c}{time share (\%)} \\", r"\cmidrule(lr){5-6}",
                r"Collection & threads & passes (s) & speedup & commit & transfer \\"], rows_t)
    right = sub("(b) Batch size $B$ (seed 42)", "lrrrrr",
                [r" & & \multicolumn{2}{c}{recall@10 (\%)} & & \\", r"\cmidrule(lr){3-4}",
                 r"Collection & $B$ & pass 1 & pass 5 & rejected (\%) & passes (s) \\"], rows_b)
    write_gen("tab_sys.tex", "\n".join([
        r"\begin{table*}[pos=!tb]", r"\centering\small", caption_tex("Parallel execution and batch size."), r"\label{tab:sys}",
        r"\setlength{\tabcolsep}{3pt}", left + r"\hfill" + right, r"\end{table*}"]) + "\n")
    if rows_b:
        rj = [float(r[4]) for r in rows_b]
        setv("sys.B.rej.max", max(rj), 2)
        d1, d5 = {}, {}
        for r in rows_b:
            d1.setdefault(r[0], []).append(float(r[2]))
            d5.setdefault(r[0], []).append(float(r[3]))
        setv("sys.B.p1.spread.max", max(max(v) - min(v) for v in d1.values()), 2)
        setv("sys.B.p5.spread.max", max(max(v) - min(v) for v in d5.values()), 2)


# ---------------------------------------------------------------- 摘要与 highlights（含数字；自动检查长度）
ABSTRACT_A = r"""Embedding retrieval serves queries from inverted-file (IVF) indexes built by $k$-means. The clustering dominates the cost of every re-index. Hierarchical $k$-means is fast, but its index loses \V{rq1.hiergap.min}--\V{rq1.hiergap.max} recall points at an equal scan budget. We propose \ours, which refines the hierarchical partition in place. Each vector is compared with only $K$ nearby centroids. Exact stability certificates prove that most vectors cannot move and skip their distance computations without changing the output. They hold under batch Lloyd and sequential, size-weighted Hartigan updates. We evaluate \ours on eight collections and MS~MARCO passage retrieval with \V{stat.seeds.main} seeds and equivalence tests. Five certified passes end at most half a point below full-data $k$-means on \V{rq1.cr5.within.count} of eight collections. They build the index \V{sp.cr5.fivf.min}--\V{sp.cr5.fivf.max} times faster than FAISS's IVF default and \V{sp.cr5.ffull.min}--\V{sp.cr5.ffull.max} times faster than full-data FAISS. The certificates skip \V{rq3.skip.p2.min}--\V{rq3.skip.p5.max}\% of later-pass candidate distances and leave the index unchanged. The cheapest certified build is faster than shortlist Lloyd refinement and FAISS's two-level clustering on every collection. Its recall is significantly lower than theirs only on \V{bl.hsltlc.L.list}. At ten million vectors, it is \V{deep.bl.vsrefine.min}--\V{deep.bl.vsrefine.max} times faster than every other refinement at equivalent recall. On MS~MARCO at a 1\% scan budget, MSCLAUSE is equivalent to five certified passes within 0.5 MRR@10 and nDCG@10 points. For top-10 effectiveness, the training method is therefore a build-cost decision. The start, not the update rule, dominates cost and quality."""
MS_CLAUSE_A = "every refined and full-data index"
MS_CLAUSE_B = "every refined index started from the exact assignment and every full-data index"

HIGHLIGHTS = [
    r"Five certified passes build an IVF index \V{sp.cr5.fivf.min}-\V{sp.cr5.fivf.max}x faster than FAISS's IVF default",
    r"Stability certificates skip \V{rq3.skip.p2.min}-\V{rq3.skip.p5.max}% of later-pass distances, never changing the index",
    r"MS MARCO: refined and full-data indexes equivalent within 0.5 MRR@10 at a 1% budget",
    r"At 10M vectors the cheapest certified build is \V{deep.bl.vsrefine.min}-\V{deep.bl.vsrefine.max}x faster than other refinements",
    r"Evaluate IVF partitions at equal scan cost: fixed-nprobe rankings can reverse",
]


def abstract_and_highlights():
    def subst(s):
        return re.sub(r"\\V\{([^}]+)\}", lambda m: V.get(m.group(1), "??" + m.group(1)), s)

    def plain_of(s):
        p = subst(s).replace(r"\ensuremath{-}", "-").replace(r"\,", " ")
        return re.sub(r"\$[^$]*\$", "k", p).replace(r"\texttt{nprobe}", "nprobe").replace("~", " ").replace(r"\%", "%")
    use_a = V.get("ms.eqall.mrr10") == "yes" and V.get("ms.eqall.ndcg10") == "yes"
    variants = {"A": ABSTRACT_A.replace("MSCLAUSE", MS_CLAUSE_A), "B": ABSTRACT_A.replace("MSCLAUSE", MS_CLAUSE_B)}
    used = "A" if use_a else "B"
    print("abstract variant used:", used, "(ms.eqall.mrr10 =", V.get("ms.eqall.mrr10"), ", ms.eqall.ndcg10 =", V.get("ms.eqall.ndcg10"), ")")
    # cas-sc 的 abstract/highlights 环境会原样收集内容，环境内不能再 \input，所以这里写出完整环境并代入数字
    write_gen("abstract.tex", "\\begin{abstract}\n" + subst(variants[used]) + "\n\\end{abstract}\n")
    for k, s in variants.items():
        write_gen(f"abstract_variant{k}.txt", plain_of(s) + "\n")
    plain = plain_of(variants[used])
    nw = len(plain.split())
    SUMMARY["abstract_words"] = nw
    SUMMARY["abstract_variant"] = used
    write_gen("abstract_plain.txt", plain + "\n")
    if nw > 245:
        print("ERROR abstract words", nw, "> 245")
    hl = []
    for h in HIGHLIGHTS:
        p = subst(h).replace(r"\ensuremath{-}", "-")
        if len(p) > 85:
            print("ERROR highlight too long:", len(p), p)
        hl.append(p)
    write_gen("highlights.tex", "\\begin{highlights}\n" + "\n".join(r"\item " + subst(h).replace("%", r"\%").replace("'", "'") for h in HIGHLIGHTS) + "\n\\end{highlights}\n")
    write_gen("highlights_plain.txt", "\n".join(hl) + "\n")
    print("abstract words:", nw, "| highlight lengths:", [len(h) for h in hl])


# ---------------------------------------------------------------- 占位：数据未到时生成占位图/表，保证论文可编译、引用不断
EXPECTED_FIGS = ["fig_pareto", "fig_scan", "fig_qps", "fig_cert", "fig_ablation", "fig_ksweep", "fig_maint", "fig_fsweep"]
EXPECTED_TABS = {"tab_main_recall": "tab:main", "tab_main_time": "tab:time", "tab_extra": "tab:extra",
                 "tab_msmarco": "tab:msmarco", "tab_wel": "tab:wel", "tab_sys": "tab:sys",
                 "tab_ablation": "tab:ablation", "tab_single": "tab:single", "tab_start": "tab:start", "tab_cscale": "tab:cscale",
                 "tab_maint": "tab:maint", "tab_protocol": "tab:protocol", "tab_final": "tab:final", "tab_fsweep": "tab:fsweep",
                 "tab_mem": "tab:mem"}


def ensure_placeholders():
    for f in EXPECTED_FIGS:
        p = os.path.join(FIG, f + ".pdf")
        if not os.path.exists(p):
            fig, ax = plt.subplots(figsize=FS(1.2))
            ax.axis("off")
            ax.text(0.5, 0.5, f"{f}: results pending", ha="center", va="center", color=MUTED)
            fig.savefig(p)
            plt.close(fig)
            print("placeholder figure", f)
    for f, lab in EXPECTED_TABS.items():
        p = os.path.join(GEN, f + ".tex")
        if not os.path.exists(p) or open(p, encoding="utf-8").read().startswith("% placeholder"):
            open(p, "w", encoding="utf-8").write(
                "% placeholder: results pending\n\\begin{table}[pos=!tb]\\centering\\caption{Results pending.}\\label{" + lab + "}\\end{table}\n")
            print("placeholder table", f)


def write_values():
    lines = ["% 由 experiments/analysis/make_assets.py 自动生成，请勿手工修改"]
    for k in sorted(V):
        lines.append(r"\defV{" + k + "}{" + V[k] + "}")
    write_gen("values.tex", "\n".join(lines) + "\n")
    SUMMARY["missing"] = MISSING
    json.dump(dict(values=V, summary=SUMMARY), open(os.path.join(GEN, "summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    md = ["# Generated keys (experiments/analysis/make_assets.py)", "", f"{len(V)} keys; use as `\\V{{key}}` in the text.", ""]
    md += [f"- `{k}` = `{V[k]}`" for k in sorted(V)]
    if MISSING:
        md += ["", "## Missing inputs (rerun after pull_results.sh)", ""] + [f"- {m}" for m in MISSING]
    write_gen("claims_keys.md", "\n".join(md) + "\n")



# ---------------------------------------------------------------- 最终列表分配的耗时（assign_time.json）与流水线各阶段耗时（图 1 / 设置节）
def stage_times():
    p = os.path.join(L.RES, "assign_time.json")
    if not os.path.exists(p):
        MISSING.append("assign_time.json")
        return
    at = json.load(open(p))
    one_m = []
    for ds, v in at.items():
        setv(f"assign.t.{ds}", v["assign_s"], 1 if v["assign_s"] < 100 else 0)
        if ds in DS and ds != "deep10m":
            one_m.append(v["assign_s"])
    if one_m:
        setv("assign.t.1m.min", min(one_m), 1)
        setv("assign.t.1m.max", max(one_m), 1)
    # 阶段耗时（种子均值）：分层构建 / 精确起步分配（含候选图与状态）/ 五轮精修，来自 cr_s*.refine.json；标签起步的五轮来自 cr_labels_s*
    for ds in list(DS) + ["msmarco"]:
        d = os.path.join(L.RES, ds)
        rows = []
        for sd in L.SEEDS:
            f = os.path.join(d, f"cr_s{sd}.refine.json")
            if os.path.exists(f):
                r = json.load(open(f))
                t, h = r["timing"], r["hist"]
                if len(h) > 5:
                    rows.append((t.get("hier_s", 0.0), t.get("init_assign_s", 0.0) + t.get("graph0_s", 0.0) + t.get("setup_s", 0.0),
                                 sum(x["epoch_s"] for x in h[1:6]), h[5]["total_s"]))
        if rows:
            a = np.mean(rows, axis=0)
            setv(f"stage.{ds}.hier.t", a[0], 1 if a[0] < 100 else 0)
            setv(f"stage.{ds}.init.t", a[1], 1 if a[1] < 100 else 0)
            setv(f"stage.{ds}.passes.t", a[2], 1 if a[2] < 100 else 0)
            setv(f"stage.{ds}.train.t", a[3], 1 if a[3] < 100 else 0)
        rows = []
        for sd in L.SEEDS:
            f = os.path.join(d, f"cr_labels_s{sd}.refine.json")
            if os.path.exists(f):
                h = json.load(open(f))["hist"]
                if len(h) > 5:
                    rows.append(sum(x["epoch_s"] for x in h[1:6]))
        if rows:
            setv(f"stage.{ds}.lpasses.t", float(np.mean(rows)), 1 if np.mean(rows) < 100 else 0)

if __name__ == "__main__":
    write_macros()
    cmp = stats_main()
    S = main_tables(cmp)
    rq1_numbers(cmp, S)
    fig_pareto()
    fig_scan()
    rq3_cert()
    mem_table()
    rq3_start_and_ablation()
    extra_table()
    protocol()
    qps()
    ksweep_cscale()
    msmarco()
    maint()
    fsweep()
    timing_flags()
    leak_numbers()
    final_assignment()
    stage_times()
    discussion_numbers()
    ablation_table()
    criterion_numbers()
    systems()
    abstract_and_highlights()
    ensure_placeholders()
    write_values()
    print(len(V), "values;", len(MISSING), "missing inputs;", "figs:", sorted(os.listdir(FIG))[-6:])
