// 实验驱动（go/no-go 实验用）。在 supercdkm_bench 的基础上增加：
//   SKM_SAMPLING=<0..1>   抽样比例（默认 1.0 全量）
//   SKM_MAXPPC=<int>      每簇最多抽样点数（默认 1<<30；SuperKMeans 原版默认 256）
//   SKM_HIER=1            用 HierarchicalSuperKMeans（原版分层版，meso 3 轮 + fine 5 轮）
//   SKM_HIER_REFINE=<int> 分层版的 refinement 轮数（默认 0）
//   SKM_INIT=<fvecs>      用给定质心初始化（原始空间），用于"分层 + CDKM/候选精修"
//   SKM_CANDK / SKM_CANDR / SKM_CANDDP / SKM_PERCAND  同 supercdkm_bench
// 用法: scdkm_exp_bench.out <base.fvecs> <k> <iters> <hartigan 0|1> <early_term 0|1>
//                           <out_prefix|-> [queries.fvecs] [n_queries]
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/superkmeans.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static std::vector<float> ReadFvecs(const std::string& path, size_t& n, size_t& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(2);
    }
    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&dim), 4);
    f.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(f.tellg());
    d = dim;
    const size_t row_bytes = 4 + d * 4;
    n = bytes / row_bytes;
    std::vector<float> out(n * d);
    f.seekg(0, std::ios::beg);
    std::vector<char> row(row_bytes);
    for (size_t i = 0; i < n; ++i) {
        f.read(row.data(), row_bytes);
        std::memcpy(out.data() + i * d, row.data() + 4, d * 4);
    }
    return out;
}
static void WriteFvecs(const std::string& path, const float* data, size_t n, size_t d) {
    std::ofstream f(path, std::ios::binary);
    const int32_t dim = static_cast<int32_t>(d);
    for (size_t i = 0; i < n; ++i) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(data + i * d), d * 4);
    }
}
static const char* Env(const char* k) { return std::getenv(k); }
static int EnvInt(const char* k, int def) { const char* v = Env(k); return v ? std::atoi(v) : def; }
static float EnvFloat(const char* k, float def) { const char* v = Env(k); return v ? std::atof(v) : def; }

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "usage: scdkm_exp_bench.out <base.fvecs> <k> <iters> <hartigan 0|1> "
                     "<early_term 0|1> <out_prefix|-> [queries.fvecs] [n_queries]\n";
        return 1;
    }
    const std::string base_path = argv[1];
    const size_t k = std::atoll(argv[2]);
    const uint32_t iters = std::atoi(argv[3]);
    const bool hartigan = std::atoi(argv[4]) != 0;
    const bool early_term = std::atoi(argv[5]) != 0;
    const std::string out_prefix = argv[6];

    const bool percand = EnvInt("SKM_PERCAND", 1) != 0;
    const uint32_t cand_k = EnvInt("SKM_CANDK", 0);
    const uint32_t cand_r = EnvInt("SKM_CANDR", 5);
    const bool cand_dp = EnvInt("SKM_CANDDP", 0) != 0;
    const float sampling = EnvFloat("SKM_SAMPLING", 1.0f);
    const uint32_t maxppc = EnvInt("SKM_MAXPPC", 1 << 30);
    const bool hier = EnvInt("SKM_HIER", 0) != 0;
    const uint32_t hier_refine = EnvInt("SKM_HIER_REFINE", 0);
    const char* init_path = Env("SKM_INIT");
    const uint32_t seed = EnvInt("SKM_SEED", 42);

    size_t n = 0, d = 0;
    auto data = ReadFvecs(base_path, n, d);
    std::vector<float> queries;
    size_t n_queries = 0;
    if (argc > 7) {
        size_t nq = 0, dq = 0;
        queries = ReadFvecs(argv[7], nq, dq);
        n_queries = (argc > 8) ? std::min<size_t>(std::atoll(argv[8]), nq) : nq;
    }
    // SKM_SAMPLE_QUERIES=N：recall 早停所用的 N 条查询从训练数据中抽样（SuperKMeans 的 sample_queries 选项），
    // 不读外部查询文件，从而不会接触评测查询。
    const uint32_t sample_q = EnvInt("SKM_SAMPLE_QUERIES", 0);
    if (sample_q > 0) {
        queries.clear();
        n_queries = sample_q;
    }
    std::vector<float> init;
    if (init_path) {
        size_t ni = 0, di = 0;
        init = ReadFvecs(init_path, ni, di);
        if (ni != k || di != d) {
            std::cerr << "SKM_INIT shape mismatch: " << ni << "x" << di << " vs " << k << "x" << d << "\n";
            return 3;
        }
    }

    std::vector<float> centroids;
    double train_ms = 0;
    std::cout << "config,hier=" << hier << ",hartigan=" << hartigan << ",candk=" << cand_k
              << ",candr=" << cand_r << ",sampling=" << sampling << ",maxppc=" << maxppc
              << ",early=" << early_term << ",iters=" << iters << ",init=" << (init_path ? 1 : 0)
              << ",nq=" << n_queries << "\n";

    if (hier) {
        skmeans::HierarchicalSuperKMeansConfig config;
        config.seed = seed;
        config.sampling_fraction = sampling;
        config.max_points_per_cluster = maxppc;
        config.early_termination = early_term;
        config.verbose = true;
        config.iters_refinement = hier_refine;
        config.use_blas_only = EnvInt("SKM_BLAS_ONLY", 0) != 0;
        if (Env("SKM_HIER_MESO")) config.iters_mesoclustering = EnvInt("SKM_HIER_MESO", 3);
        if (Env("SKM_HIER_FINE")) config.iters_fineclustering = EnvInt("SKM_HIER_FINE", 5);
        skmeans::HierarchicalSuperKMeans kmeans(k, d, config);
        auto t0 = std::chrono::steady_clock::now();
        centroids = kmeans.Train(data.data(), n);
        auto t1 = std::chrono::steady_clock::now();
        train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // SKM_SAVE_LABELS=1：导出分层版训练结束时的点分配（meso 内分配，训练的副产物，不额外计算），
        // 供 CertRefine 直接从分层标签起步，避免再做一次 n×c 全量分配。仅在未抽样时顺序与原数据一致。
        if (EnvInt("SKM_SAVE_LABELS", 0) != 0 && out_prefix != "-") {
            if (sampling < 1.0f) {
                std::cerr << "SKM_SAVE_LABELS requires SKM_SAMPLING=1 (labels would be in sampled order)\n";
            } else {
                std::ofstream lf(out_prefix + ".labels.u32", std::ios::binary);
                lf.write(reinterpret_cast<const char*>(kmeans.assignments.get()), sizeof(uint32_t) * n);
            }
        }
        std::cout << "iter,objective,shift,split,recall,not_pruned_pct,partial_d,gemm_only,moves\n";
        const auto& hs = kmeans.hierarchical_iteration_stats;
        size_t it = 0;
        for (const auto* vec : {&hs.mesoclustering_iteration_stats, &hs.fineclustering_iteration_stats,
                                &hs.refinement_iteration_stats}) {
            for (const auto& s : *vec) {
                std::cout << (++it) << "," << s.objective << "," << s.shift << "," << s.split << ","
                          << s.recall << "," << s.not_pruned_pct << "," << s.partial_d << ","
                          << (s.is_gemm_only ? 1 : 0) << ",0\n";
            }
        }
    } else {
        skmeans::SuperKMeansConfig config;
        config.iters = iters;
        config.seed = seed;
        config.sampling_fraction = sampling;
        config.max_points_per_cluster = maxppc;
        config.early_termination = early_term;
        config.verbose = true;
        config.hartigan = hartigan;
        config.hartigan_percand = percand;
        config.candidate_k = cand_k;
        config.candidate_refresh = cand_r;
        config.candidate_dimprune = cand_dp;
        // SKM_BLAS_ONLY=1：关闭 ADSampling 维度剪枝，每轮都用精确 GEMM 分配（排查剪枝对质量的影响）
        config.use_blas_only = EnvInt("SKM_BLAS_ONLY", 0) != 0;
        if (!init.empty()) config.initial_centroids = init.data();
        config.sample_queries = sample_q > 0;
        skmeans::SuperKMeans kmeans(k, d, config);
        auto t0 = std::chrono::steady_clock::now();
        centroids = kmeans.Train(data.data(), n, (n_queries && !queries.empty()) ? queries.data() : nullptr, n_queries);
        auto t1 = std::chrono::steady_clock::now();
        train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "iter,objective,shift,split,recall,not_pruned_pct,partial_d,gemm_only,moves\n";
        for (const auto& s : kmeans.iteration_stats) {
            std::cout << s.iteration << "," << s.objective << "," << s.shift << "," << s.split << ","
                      << s.recall << "," << s.not_pruned_pct << "," << s.partial_d << ","
                      << (s.is_gemm_only ? 1 : 0) << "," << s.hartigan_moves << "\n";
        }
    }
    std::cout << "train_ms," << train_ms << "\n";
    if (out_prefix != "-") {
        WriteFvecs(out_prefix + ".centroids.fvecs", centroids.data(), k, d);
    }
    return 0;
}
