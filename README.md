# CertRefine: code, run logs and results

This repository accompanies the paper **"Fast clustering for inverted-file vector indexes with stability certificates"**.
It contains three things:

- the programs that ran every experiment;
- the logs and result files of all runs;
- the analysis code that turns these files into every number, table and figure of the paper.

You can regenerate all results of the paper from the logs without downloading any vector data (Section 1).
Sections 2 and 3 map each part of the paper to the logs behind it. Section 5 describes how to rerun the experiments.

Table, figure and section numbers refer to the submitted manuscript and its supplementary material.

## Contents

| Path | Contents |
|---|---|
| `experiments/` | Experiment drivers (`refine.py`, `faiss_km.py`, `run_cpp.sh`, ...), the experiment chains that ran them (`chain_*.sh`), the MS MARCO preparation (`msmarco/`) and helper tools (`tools/`) |
| `experiments/results/` | Logs and result files of all runs: 8,976 files, including 970 training logs |
| `experiments/analysis/` | `make_assets.py` and `lib.py`: statistics, tables, figures and every number of the text |
| `SuperCDKM/` | The C++ engine: SuperKMeans with our additions and the C++ port of CertRefine (Section 6) |
| `checks/` | Two scripts that check the results and logs |

## 1. Regenerate the results of the paper

You need Python 3 with numpy and matplotlib. The results of the paper were produced with Python 3.13.9, numpy 2.3.5 and matplotlib 3.10.8.
Other matplotlib versions may change individual pixels of the figures.

```bash
python experiments/analysis/make_assets.py   # about 3 minutes
python checks/check_results.py               # a few seconds
```

`make_assets.py` reads `experiments/results/` and writes everything the paper takes from the experiments into `paper/`.
Git ignores this folder.

- `paper/generated/values.tex`: every number of the text (Section 2.1);
- `paper/generated/tab_*.tex`: the tables;
- `paper/figs/*.pdf`: the figures.

The paper typesets some tables in another layout with the same numbers. It draws Figures 3, 4 and 6 from the same data in another arrangement of panels.

`checks/check_regeneration.py` runs the same generation in a temporary folder and reports any error. Before release, we compared its output with the files of the submitted paper:

- the numbers and tables were identical;
- the figures that the paper uses as drawn were identical pixel by pixel.

`checks/check_results.py` runs 55 checks of the result files and logs against the paper:

- the statistics of Table 1;
- the seeds of every configuration;
- that every log is complete and free of errors;
- the settings and seeds of every sweep and side study;
- that the superseded runs are kept apart from the reported ones.

Two inputs are reported as missing on purpose, and the paper states both gaps:

- the candidate-count sweep was not run on Deep-10M;
- FAISS-sampled has three seeds on MS MARCO.

## 2. From the paper to the logs

### 2.1 Numbers in the text

Every experimental number in the text and in the supplement is a data key defined in `paper/generated/values.tex` (7,744 keys).
`make_assets.py` writes each key with `setv()`, so searching that file for the prefix of a key finds the code that computes it.

- `paper/generated/claims_keys.md` lists all keys with their values.
- `paper/generated/summary.json` holds the same values together with the underlying statistics: differences, bootstrap intervals and Holm-adjusted p-values.

| Key prefix | Content | Function in `make_assets.py` |
|---|---|---|
| `ds.<collection>.*`, `sd.*` | Per-collection means and standard deviations of build time and recall | `main_tables`, `rq1_numbers` |
| `wel.*` | Pairwise comparisons: significance, equivalence, speedups (Table 5) | `stats_main` |
| `sp.*`, `rq1.*`, `pareto.*`, `bl.*`, `deep.*`, `bal.*`, and per-method prefixes such as `cr1.*`, `hor1.*`, `ws2.*`, `tlc.*`, `hslskm.*` | Speedups and comparisons of Section 5.2 | `rq1_numbers` (through `speed_keys` and `pair_range`) |
| `ms.*` | MS MARCO (Section 5.3, Table 6) | `msmarco` |
| `rq3.*`, `psweep.*`, `ceil.*`, `move.*`, `bd.*`, `fixpt.*` | Certificates, single bound, cost breakdown (Section 5.4) | `rq3_cert` |
| `mem.*` | Peak memory (Section 5.7, Table S4) | `rq3_cert` |
| `start.*` | Starting assignments (Table 8) | `rq3_start_and_ablation` |
| `abl.*`, `crit.*` | Criterion and update rule (Table 9) | `ablation_table`, `criterion_numbers` |
| `k.*`, `c.*` | Candidates, passes and number of lists (Figure 7, Table 10) | `ksweep_cscale` |
| `sys.*` | Threads and batch size (Table S3) | `systems` |
| `fsweep.*` | FAISS tuned per collection (Figure 4, Table 11) | `fsweep` |
| `bcdkm.*`, `mb.*`, `k2.*` | Additional baselines (Table 12) | `extra_table` |
| `maint.*` | Maintenance (Section 5.8, Table 13) | `maint` |
| `prot.*` | Fixed nprobe versus equal scan (Section 5.9, Table 14) | `protocol` |
| `fin.*` | Final-assignment variants (Table S2) | `final_assignment` |
| `qps.*` | Query throughput (Figure S1) | `qps` |
| `stage.*`, `assign.*` | Stage times and the time of the final exact assignment | `stage_times` |
| `ret.*`, `leak.*` | Re-timed runs and superseded runs (Tables S6 and S7) | `timing_flags`, `leak_numbers` |
| `disc.*`, `cpp.*`, `color.*`, `cr5.*` | Discussion, C++ port, conflict coloring | `discussion_numbers` |
| `stat.*`, `fam.*` | Numbers of seeds, method families | several |

### 2.2 Tables and figures

The functions are in `experiments/analysis/make_assets.py`, and the output files are in `paper/generated/` and `paper/figs/`.
Result files are in `experiments/results/<collection>/`, and their tags are explained in Section 3.

| Paper element | Function → output | Result files |
|---|---|---|
| Table 1 (collections) | Written in the paper. `check_results.py` checks it against the result files | `cr_s42.refine.json` (n, d, c), `experiments/data_msmarco/msmarco_meta.json` |
| Table 2 (methods) | Written in the paper | Section 3 below |
| Table 3 (recall) and Table 4 (build time) | `main_tables` → `tab_main_recall.tex`, `tab_main_time.tex` | `<tag>_s<seed>.scan.json`, `.log`, `.refine.json`, and `results/assign_time.json` (final assignment) |
| Table 5 (pairwise comparisons) | `stats_main`, `rq1_numbers` → `tab_wel.tex` | As Tables 3 and 4, plus the per-query recall `.pqb.npy` |
| Figure 3 (build time versus recall) | `fig_pareto` → `fig_pareto.pdf` | As Tables 3 and 4 |
| Figure 4 (FAISS over iterations and sample sizes) | `fsweep` → `fig_fsweep.pdf` | `faiss_it<I>_m<M>`, `faiss_it<I>_full`, `faiss_ivf`, `faiss` (and `faiss_sampled` on Deep-10M) |
| Figure 5 (recall difference across scan budgets) | `fig_scan` → `fig_scan.pdf` | `.scan.json` (recall against the scanned fraction) |
| Table 6 (MS MARCO) | `msmarco` → `tab_msmarco.tex` | `msmarco/<tag>_s<seed>[_ep<P>].ir.json`, `.ir_pqb.npz`, `.log`, `.refine.json` |
| Figure 6 (work removed by the certificates) | `rq3_cert` → `fig_cert.pdf` | `cr_s42.refine.json`, `cr_nocert_s42.refine.json` |
| Table 7 (one bound versus K bounds) | `rq3_cert` → `tab_single.tex` | `single_s42.refine.json`, `cr_s42.refine.json` |
| Table 8 (starting assignment) | `rq3_start_and_ablation` → `tab_start.tex` | `cr_labels`, `cr_snap`, `cr_greedy`, `cr_ivfgemm`, `cr_ann` (`diag` field of `.refine.json`) |
| Table 9 (criterion and update rule) | `ablation_table`, `criterion_numbers` → `tab_ablation.tex` | `cr`, `sl`, `hb`, `hslc` after passes 1 and 5 (`_ep1`, `_ep5`) |
| Figure 7 (candidates and passes), Table 10 (lists) | `ksweep_cscale` → `fig_ksweep.pdf`, `tab_cscale.tex` | `cr_K<K>`; folders `sift_c1000/`, `sift_c2000/`, `sift_c8000/` |
| Table 11 (cheapest FAISS at equal recall) | `fsweep` → `tab_fsweep.tex` | As Figure 4 |
| Table 12 (additional baselines) | `extra_table` → `tab_extra.tex` | `bcdkm`, `bcdkm_samp`, `faiss_sampled`, `lloyd_exact`, `sk_elkan`, `sk_minibatch`, `k2style` |
| Table 13 (maintenance) | `maint` → `tab_maint.tex` | `maint/<workload>_s<seed>.json`, `maint/<workload>_s42_control.json` |
| Table 14 (fixed nprobe versus equal scan) | `protocol` → `tab_protocol.tex` | `.scan.json` (recall at every nprobe and at the 1% budget) |
| Table S2 (final-assignment variants) | `final_assignment` → `tab_final.tex` | `crf`, `crf_labels`, `bf_labels` |
| Table S3 (threads and batch size) | `systems` → `tab_sys.tex` | `cr_thr<T>_s42`, `cr_B<B>_s42` |
| Table S4 (peak memory) | `mem_table` → `tab_mem.tex` | `results/peakmem/<collection>_<tag>.txt` |
| Figure S1 (query throughput) | `qps` → `fig_qps.pdf` | `<tag>_s42*.qps.json` |
| Figure S2 (recall after each insertion batch) | `maint` → `fig_maint.pdf` | `maint/*.json` |
| Table S6 (re-timed runs) | `timing_flags` → `timing_deviations.txt`. The rule is `experiments/tools/timing_outliers.py` | `.log`, `.refine.json` |
| Table S7 (superseded SKM-default runs) | `leak_numbers` | `<collection>/_superseded/` |
| Table S8 (method names and tags) | Written in the supplement | Section 3 below |

`fig_ablation.pdf` is drawn by `rq3_start_and_ablation` but not used in the paper.
The files `abstract*` and `highlights*` in `paper/generated/` are by-products of an earlier draft, and the paper does not use them.

## 3. Methods and their result files

Each run writes `experiments/results/<collection>/<tag>_s<seed>.<ext>`.
For the refinements, rows after pass P carry the suffix `_ep<P>`.
The main comparison uses seeds 42, 1, 2, 3 and 4. Most sweeps use seeds 42, 1 and 2.

| Name in the paper | Tag | Script and options |
|---|---|---|
| Hier-SKM | `hier` | `run_cpp.sh hier`: SuperKMeans hierarchical build, which also exports its labels |
| FAISS-2L | `faiss_2lc` | `faiss_2lc.py`: FAISS `two_level_clustering`, round(√c) first-level clusters, rebalancing |
| CertRefine-H-L | `cr_labels` | `refine.py --init labels`: Hartigan, sequential, certificates, 5 passes |
| CertRefine-H-1, CertRefine-H-5 | `cr` (`_ep1`, `_ep5`) | `refine.py`: exact start, rows after pass 1 and pass 5 |
| CertRefine-B-L | `hslc_labels` | `refine.py --rule lloyd --update batch --cert 1 --init labels` |
| CertRefine-B-5 | `hslc` | `refine.py --rule lloyd --update batch --cert 1` |
| HSL (no cert) | `hsl` | `refine.py --rule lloyd --update batch --cert 0` |
| SL-5 | `sl` | `refine.py --rule lloyd --update seq --cert 1` |
| CertRefine-B-L (R=1) | `k2style` | `refine.py --rule lloyd --update batch --cert 1 --init labels --refresh 1` |
| HSL-SKM | `hsl_cpp` | `run_cpp.sh hsl_cpp`: our shortlist Lloyd in the C++ engine |
| HOR-1, HOR-2, HOR-5 | `hor1`, `hor2`, `hor` | `run_cpp.sh`: hierarchical build plus 1, 2 or 5 SuperKMeans refinement iterations |
| Lloyd-WS-1, -2, -3 | `warm1`, `warm2`, `warm3` | `faiss_km.py --init hier --niter k` |
| SKM-default | `skm_default` | `run_cpp.sh skm_default`: SuperKMeans defaults, early stopping on 1,000 queries sampled from the training data |
| FAISS-IVF | `faiss_ivf` | `faiss_km.py --niter 10 --mpc 256` |
| FAISS-full | `faiss` | `faiss_km.py`: 25 iterations on the full collection |
| FAISS-sampled | `faiss_sampled` | `faiss_km.py --mpc 256`: 25 iterations |
| FAISS sweep | `faiss_it<I>_m<M>`, `faiss_it<I>_full` | `faiss_km.py --niter I` with `--mpc M` or the full collection |
| SKM-Lloyd | `lloyd` | `run_cpp.sh lloyd` |
| SKM-Lloyd, no pruning | `lloyd_exact` | `run_cpp.sh lloyd_exact` |
| Batched CDKM | `bcdkm` | `run_cpp.sh bcdkm`: our earlier batched variant of CDKM (supplement, Section S5) |
| Batched CDKM (30% sample) | `bcdkm_samp` | The same engine on a 30% sample, seed 42, imported by `import_old.sh` |
| scikit-learn Elkan | `sk_elkan` | scikit-learn `KMeans(algorithm="elkan")`, random initialization, 25 iterations |
| MiniBatch | `sk_minibatch` | `sk_minibatch.py`: scikit-learn `MiniBatchKMeans` |
| Single bound (Table 7) | `single` | `refine_single.py` |
| Candidate count K (Figure 7) | `cr_K<K>` | `refine.py --K K` |
| Batch size B (Table S3) | `cr_B<B>` | `refine.py --T B` |
| Threads (Table S3) | `cr_thr<T>` | `refine.py --threads T` |
| Starting assignments (Table 8) | `cr_snap`, `cr_labels`, `cr_greedy`, `cr_ivfgemm`, `cr_ann` | `refine.py --init snap / labels / greedy / ivfgemm / ann --diag-exact` |
| Batch Hartigan (Table 9) | `hb` | `refine.py --rule hartigan --update batch --cert 1` |
| No certificates (Figure 6) | `cr_nocert` | `refine.py --cert 0` |
| Final-assignment variants (Table S2) | `crf`, `crf_labels`, `bf_labels` | `refine.py` with saved labels and a timed candidate-restricted assignment |
| C++ port of CertRefine | `xcr`, `xcr_nocert` | `SuperCDKM/examples/certrefine.cpp` |
| Query throughput (Figure S1) | `<tag>_s42*.qps.json` | `eval_qps.py` |
| Maintenance (Section 5.8) | `maint/<workload>_s<seed>.json`, `_control` | `maint.py`, `maint.py --control` |

The options of every refinement run are also recorded in its `.refine.json` (`rule`, `update`, `cert`, `init`, `K`, `T`, `refresh`, `seed`, `threads`).
The C++ and FAISS runs record theirs in the first line of the `.log`.

## 4. Result files

| File | Contents |
|---|---|
| `<tag>_s<seed>.log` | Training log. FAISS and the C++ engine write a `config,...` line and `train_ms,<milliseconds>` (training only). Warm starts also write `hier_ms` and `lloyd_ms`. The C++ engine adds one line per iteration |
| `<tag>_s<seed>.refine.json` | One refinement run: options, `timing` (hierarchical build `hier_s`, candidate graph `graph0_s`, starting assignment `init_assign_s`, `setup_s`, in seconds), and `hist` with one entry per pass (`moves`, `rejected`, `pairs_total`, `pairs_skipped`, `skipped_pct`, `pass_s`, `epoch_s`, `total_s`, `labels_md5`). `diag` holds the agreement of a starting assignment with the exact one |
| `<tag>_s<seed>[_ep<P>].scan.json` | Equal-scan evaluation: recall@10 at every nprobe from 1 to min(c, 1024) (`nprobes`, `scan_frac`, `recall`), recall at scan budgets of 0.25, 0.5, 1, 2 and 4% (`recall@scan0.01`, ...), list statistics (`cv`, `max_list`, `empty`) and the objective `wcss` |
| `.pqb.npy` | Per-query recall@10 at the five budgets (float32, queries × 5). Input of the paired bootstrap |
| `.hits.npy` | Per-query hits on a coarse grid of 16 nprobe values (uint8). Kept for compatibility with the earlier CPU evaluator |
| `msmarco/*.ir.json` | MS MARCO retrieval: MRR@10, nDCG@10, Recall@100 and ANN recall@10 at each budget, plus the dense curves |
| `msmarco/*.ir_pqb.npz` | Per-query values of these four metrics at the five budgets |
| `*.qps.json` | Queries per second against recall at 20 threads |
| `maint/<workload>_s<seed>.json` | One maintenance run: build, every insertion batch and the rebuild for frozen lists, local Lloyd, certified Lloyd and certified Hartigan. The `.pqb.npy` files beside it hold the per-query recall. `maint/_v1/` holds an earlier seed-42 version without per-query output, which the paper does not use |
| `peakmem/<collection>_<tag>.txt` | Report of GNU `time -v` (peak resident memory) |
| `assign_time.json` | Time of the final exact assignment of all vectors per collection (three repetitions, median used) |
| `coloring.json` | Greedy coloring of the conflict graph (discussion) |
| `<collection>/_superseded/` | SKM-default runs whose early stopping used the evaluation queries (Table S7). Not used in the paper |
| `IMPORTED_FROM_2026-09-22.txt` | Provenance of the seed-42 flat baselines imported by `import_old.sh` from an earlier round on the same machine |
| `.core_done`, `.train_done`, ... | Completion markers of the experiment chains |

Build time in the paper is end-to-end:

- **Included:** training (for the refinements, also the hierarchical build, the candidate graph and the starting assignment) and the final exact assignment from `assign_time.json`.
- **Excluded:** reading the data and JIT compilation.

All timed runs used 20 threads on one NUMA node and ran one at a time. Evaluation was not timed and ran on the other node and on the GPUs.

## 5. Rerunning the experiments

### Collections

The vector data are not included. `experiments/common.py` (`DATASETS`) lists the expected files: `<name>_base.fvecs`, `<name>_query.fvecs` and `<name>_gt.ivecs` under `$SCDKM_DATA`, `$DATA10M` or `$DATA_MSMARCO`.
The evaluation uses the first 2,000 queries, or all of them when there are fewer.

| Collection | n | d | c | Queries | Source |
|---|---|---|---|---|---|
| SIFT | 1,000,000 | 128 | 4,000 | 2,000 | ANN-Benchmarks `sift-128-euclidean` |
| GloVe | 1,183,514 | 200 | 4,352 | 2,000 | ANN-Benchmarks `glove-200-angular`, ℓ2-normalized |
| Yahoo | 677,305 | 384 | 3,292 | 1,000 | VIBE `yahoo-minilm-384-normalized` |
| GIST | 1,000,000 | 960 | 4,000 | 1,000 | ANN-Benchmarks `gist-960-euclidean` |
| MXBAI | 769,382 | 1024 | 3,509 | 1,000 | VIBE `agnews-mxbai-1024-euclidean` |
| DBpedia | 229,772 | 1536 | 1,917 | 1,000 | Hugging Face `KShivendu/dbpedia-entities-openai-1M`, first 6 of 26 shards (230,772 vectors), 1,000 held out as queries |
| Wiki | 260,372 | 3072 | 2,041 | 1,000 | VIBE `simplewiki-openai-3072-normalized` (first release, arXiv:2505.17810v1) |
| Deep-10M | 9,990,000 | 96 | 12,649 | 2,000 | ANN-Benchmarks `deep-image-96-angular`, ℓ2-normalized; c from the nominal size of 10⁷ |
| MS MARCO | 8,841,823 | 384 | 11,894 | 6,980 | `BeIR/msmarco` corpus and dev queries, `BeIR/msmarco-qrels`, encoded with BGE-small-en-v1.5 by `msmarco/chain_msmarco.sh` |

The number of lists is c = round(4√n).

### Environment

| Item | Value |
|---|---|
| Machine | 2× Intel Xeon Gold 5218R (20 cores each, two NUMA nodes), 125 GB memory, 2× RTX 3090 (only for evaluation and encoding, untimed) |
| Python for the experiments | numpy 2.4.6, numba 0.67.0, faiss-cpu 1.14.3, scikit-learn 1.9.1 |
| Python for encoding MS MARCO | torch 2.9.1 (CUDA 12.8), sentence-transformers 6.1.0, pyarrow 22.0.0 |
| C++ | GCC 11.4, `-O3 -march=native`, OpenBLAS, CMake ≥ 3.26 |

Build the engine with:

```bash
cmake -S SuperCDKM -B SuperCDKM/build -DCMAKE_BUILD_TYPE=Release
cmake --build SuperCDKM/build --target scdkm_exp_bench.out certrefine.out
```

Edit the paths at the top of `experiments/env.sh`, then source it. `source experiments/env.sh [node]` sets the data paths, the Python interpreters, the NUMA pinning (`$PIN`) and 20 threads.

### Experiment chains

The chains are restartable, and finished runs are skipped. They ran in this order:

| Chain | Runs |
|---|---|
| `chain_eval_gpu.sh`, `chain_msmarco_eval_gpu.sh` | Evaluation daemons on the second node and the GPUs: evaluate every new set of centroids at equal scan cost (untimed). `chain_eval.sh` is the earlier CPU version |
| `msmarco/chain_msmarco.sh` | MS MARCO download, encoding on two GPUs, exact top-100 and retrieval evaluation (about 1 h of encoding) |
| `chain_main.sh` | Main timed runs on node 0: stages `core`, `flat`, `cscale`, `ksweep` (about 1.5 h) |
| `chain_extra.sh`, `chain_post.sh` | Re-timing, single bound, starts from labels, Voronoi diagnostics, C++ port, MS MARCO training, maintenance, throughput, Deep-10M flat baselines |
| `chain_rev.sh`, `chain_rev_post.sh` | Leakage-free SKM-default, FAISS-IVF, HOR-1/2, Lloyd-WS, final-assignment timing, threads and batch size, seeds 3 and 4, MS MARCO baselines |
| `chain_rev2.sh`, `chain_rev2b.sh`, `chain_rev2c.sh` | More seeds for MS MARCO and the sensitivity studies, tuned FAISS, maintenance controls, approximate starts, MiniBatch, CertRefine-B-L (R=1), final-assignment variants, FAISS two-level clustering, peak memory |

`import_old.sh` imported the seed-42 flat baselines of an earlier round on the same machine, with the same binaries and FAISS version. These runs were evaluated again.

Before the five-seed runs, the rule in `tools/timing_outliers.py` flagged four runs of the three-seed results whose time deviated by more than 15% from the median of the other seeds. The `retime` stage of `chain_extra.sh` trained them again with the same seed, and only the repeated time is reported (Table S6). `tools/retime_check.sh` repeats such runs in a scratch folder to test whether a deviation reproduces.

## 6. Notes

- **C++ engine.** `SuperCDKM/` is a fork of [SuperKMeans](https://github.com/cwida/SuperKMeans) (MIT License, `SuperCDKM/LICENSE`). Our additions are:
  - the Hartigan criterion and the candidate restriction in `include/superkmeans/`, behind two switches that are off by default;
  - the experiment driver `examples/scdkm_exp_bench.cpp`, with `examples/supercdkm_bench.cpp`;
  - the C++ port of CertRefine, `examples/certrefine.cpp`.

  With both switches off, the engine runs the upstream code path.
- **Fonts.** The figures use the Roboto fonts in `experiments/analysis/fonts/` (Apache License 2.0, `fonts/LICENSE.txt`). `otf2ttf.py` there converts the OpenType fonts of TeX Live to TrueType.
- **Not included.** Centroids, labels and indexes are not included because of their size, so evaluating a run again requires training it again. The scripts that converted the ANN-Benchmarks, VIBE and DBpedia files to `fvecs` are not included either; the expected layout is given in Section 5.
- **Paths.** Server paths in scripts and logs are replaced with placeholders such as `/path/to/workspace`.
- **Comments.** Comments in the code are partly in Chinese.
