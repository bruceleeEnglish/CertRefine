// SuperCDKM head-to-head benchmark driver.
// 用法: supercdkm_bench.out <base.fvecs> <k> <iters> <hartigan 0|1> <early_term 0|1>
//                          <out_prefix|-> [queries.fvecs] [n_queries]
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
static void WriteIvecs(const std::string& path, const uint32_t* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    const int32_t dim = 1; // 每条记录一个邻居 id
    for (size_t i = 0; i < n; ++i) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(data + i), 4);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "usage: supercdkm_bench.out <base.fvecs> <k> <iters> <hartigan 0|1> "
                     "<early_term 0|1> <out_prefix|-> [queries.fvecs] [n_queries]\n";
        return 1;
    }
    const std::string base_path = argv[1];
    const size_t k = std::atoll(argv[2]);
    const uint32_t iters = std::atoi(argv[3]);
    const bool hartigan = std::atoi(argv[4]) != 0;
    const bool early_term = std::atoi(argv[5]) != 0;
    const std::string out_prefix = argv[6];

    // 环境变量开关
    const char* pc = std::getenv("SKM_PERCAND");
    const bool percand = (pc == nullptr) || (std::atoi(pc) != 0);
    const char* ck = std::getenv("SKM_CANDK");
    const uint32_t cand_k = ck ? static_cast<uint32_t>(std::atoi(ck)) : 0;
    const char* cr = std::getenv("SKM_CANDR");
    const uint32_t cand_r = cr ? static_cast<uint32_t>(std::atoi(cr)) : 5;
    const char* cdp = std::getenv("SKM_CANDDP");
    const bool cand_dp = cdp && (std::atoi(cdp) != 0);

    size_t n = 0, d = 0;
    auto data = ReadFvecs(base_path, n, d);
    std::vector<float> queries;
    size_t n_queries = 0;
    if (argc > 7) {
        size_t nq = 0, dq = 0;
        queries = ReadFvecs(argv[7], nq, dq);
        n_queries = (argc > 8) ? std::min<size_t>(std::atoll(argv[8]), nq) : nq;
    }

    skmeans::SuperKMeansConfig config;
    config.iters = iters;
    config.seed = 42;
    config.sampling_fraction = 1.0f;
    config.max_points_per_cluster = 1 << 30; // 不抽样，全量
    config.early_termination = early_term;
    config.verbose = true;
    config.hartigan = hartigan;
    config.hartigan_percand = percand;
    config.candidate_k = cand_k;
    config.candidate_refresh = cand_r;
    config.candidate_dimprune = cand_dp;

    skmeans::SuperKMeans kmeans(k, d, config);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> centroids =
        kmeans.Train(data.data(), n, n_queries ? queries.data() : nullptr, n_queries);
    auto t1 = std::chrono::steady_clock::now();
    const double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 逐迭代 CSV
    std::cout << "iter,objective,shift,split,recall,not_pruned_pct,partial_d,gemm_only,moves\n";
    for (const auto& s : kmeans.iteration_stats) {
        std::cout << s.iteration << "," << s.objective << "," << s.shift << "," << s.split << ","
                  << s.recall << "," << s.not_pruned_pct << "," << s.partial_d << ","
                  << (s.is_gemm_only ? 1 : 0) << "," << s.hartigan_moves << "\n";
    }
    std::cout << "train_ms," << train_ms << "\n";

    if (out_prefix != "-") {
        WriteFvecs(out_prefix + ".centroids.fvecs", centroids.data(), k, d);
        auto assign = kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, k);
        WriteIvecs(out_prefix + ".labels.ivecs", assign.data(), n);
    }
    return 0;
}
