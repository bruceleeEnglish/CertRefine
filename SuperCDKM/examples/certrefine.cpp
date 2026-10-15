// CertRefine (C++): certified sequential Hartigan refinement of a hierarchical k-means partition, and its
// controlled variants (Lloyd criterion, batch updates, no certificates). Port of experiments/refine.py.
//
// usage: certrefine.out <base.fvecs> <hier_prefix> <c> <out_prefix>
//   <hier_prefix>.centroids.fvecs / .labels.u32 / .log  come from scdkm_exp_bench.out (SKM_HIER=1 SKM_SAVE_LABELS=1)
// env: CR_K=64 CR_P=5 CR_B=4096 CR_SEED=42 CR_RULE=hartigan|lloyd CR_UPDATE=seq|batch CR_CERT=1
//      CR_INIT=exact|labels|snap  CR_SAVE=all|ends|none
// output: <out_prefix>_ep<e>.centroids.fvecs ; <out_prefix>.refine.json
// Timing (total_s): hierarchical build (train_ms of <hier_prefix>.log) + candidate graph + initial assignment
//   + state setup + every pass (bound transfer, permutation, parallel decide, serial commit/apply).
//   Reading files is excluded.
#include "superkmeans/common.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static const double MARGIN = 1e-4;  // relative slack subtracted whenever a bound is stored (covers float rounding)
static const double TIE = 1e-7;     // a candidate must improve the key by this relative amount to be chosen

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }
static const char* Env(const char* k, const char* d) { const char* v = std::getenv(k); return v ? v : d; }
static int EnvInt(const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; }

static std::vector<float> ReadFvecs(const std::string& path, size_t& n, size_t& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&dim), 4);
    f.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(f.tellg());
    d = dim;
    n = bytes / (4 + d * 4);
    std::vector<float> out(n * d);
    f.seekg(0, std::ios::beg);
    std::vector<char> row(4 + d * 4);
    for (size_t i = 0; i < n; ++i) {
        f.read(row.data(), row.size());
        std::memcpy(out.data() + i * d, row.data() + 4, d * 4);
    }
    return out;
}
static void WriteFvecs(const std::string& path, const float* x, size_t n, size_t d) {
    std::ofstream f(path, std::ios::binary);
    const int32_t dim = static_cast<int32_t>(d);
    for (size_t i = 0; i < n; ++i) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(x + i * d), d * 4);
    }
}
static double TrainSeconds(const std::string& log) {
    std::ifstream f(log);
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("train_ms,", 0) == 0) return std::atof(line.c_str() + 9) / 1000.0;
    return 0.0;
}
static uint64_t Fnv1a(const int32_t* a, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(a);
    for (size_t i = 0; i < n * 4; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static inline float L2sq(const float* SKM_RESTRICT a, const float* SKM_RESTRICT b, size_t d) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (size_t j = 0; j < d; ++j) {
        const float u = a[j] - b[j];
        s += u * u;
    }
    return s;
}

// C (rows x c, row-major) = A (rows x d, row-major) * B^T (B: c x d, row-major), via column-major sgemm_.
static void GemmABt(const float* A, size_t rows, const float* B, size_t c, size_t d, float* C) {
    char ta = 'T', tb = 'N';
    int m = (int)c, n = (int)rows, k = (int)d, lda = (int)d, ldb = (int)d, ldc = (int)c;
    float alpha = 1.f, beta = 0.f;
    sgemm_(&ta, &tb, &m, &n, &k, &alpha, B, &lda, A, &ldb, &beta, C, &ldc);
}

// Exact nearest-centroid assignment (blocked GEMM + argmin), as FAISS IndexFlatL2 does.
static void ExactAssign(const float* X, size_t n, size_t d, const float* M, size_t c, int32_t* out) {
    std::vector<float> mn(c);
    for (size_t k = 0; k < c; ++k) {
        double s = 0;
        for (size_t j = 0; j < d; ++j) s += (double)M[k * d + j] * M[k * d + j];
        mn[k] = (float)s;
    }
    const size_t BS = 4096;
    std::vector<float> G(BS * c);
    for (size_t b0 = 0; b0 < n; b0 += BS) {
        const size_t rows = std::min(BS, n - b0);
        GemmABt(X + b0 * d, rows, M, c, d, G.data());
#pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r) {
            const float* g = G.data() + r * c;
            float best = mn[0] - 2.f * g[0];
            int32_t bi = 0;
            for (size_t k = 1; k < c; ++k) {
                const float v = mn[k] - 2.f * g[k];
                if (v < best) { best = v; bi = (int32_t)k; }
            }
            out[b0 + r] = bi;
        }
    }
}

// K nearest centroids of every centroid (excluding itself).
static std::vector<int32_t> BuildGraph(const float* M, size_t c, size_t d, size_t K) {
    std::vector<float> G(c * c);
    GemmABt(M, c, M, c, d, G.data());
    std::vector<float> nrm(c);
    for (size_t k = 0; k < c; ++k) nrm[k] = G[k * c + k];
    std::vector<int32_t> nb(c * K);
#pragma omp parallel
    {
        std::vector<int32_t> idx(c);
        std::vector<float> dist(c);
#pragma omp for schedule(static)
        for (size_t p = 0; p < c; ++p) {
            for (size_t k = 0; k < c; ++k) dist[k] = (k == p) ? INFINITY : nrm[p] + nrm[k] - 2.f * G[p * c + k];
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                              [&](int32_t a, int32_t b) { return dist[a] < dist[b] || (dist[a] == dist[b] && a < b); });
            for (size_t j = 0; j < K; ++j) nb[p * K + j] = idx[j];
        }
    }
    return nb;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: certrefine.out <base.fvecs> <hier_prefix> <c> <out_prefix>\n";
        return 1;
    }
    const std::string base_path = argv[1], hp = argv[2], outp = argv[4];
    const size_t c = std::atoll(argv[3]);
    const size_t K = EnvInt("CR_K", 64), P = EnvInt("CR_P", 5), B = EnvInt("CR_B", 4096);
    const uint64_t seed = EnvInt("CR_SEED", 42);
    const bool lloyd = std::string(Env("CR_RULE", "hartigan")) == "lloyd";
    const bool batch = std::string(Env("CR_UPDATE", "seq")) == "batch";
    const bool use_cert = EnvInt("CR_CERT", 1) != 0;
    const std::string init = Env("CR_INIT", "exact"), save = Env("CR_SAVE", "all");

    size_t n = 0, d = 0, c0 = 0, d0 = 0;
    std::vector<float> X = ReadFvecs(base_path, n, d);
    std::vector<float> M0 = ReadFvecs(hp + ".centroids.fvecs", c0, d0);
    if (c0 != c || d0 != d) { std::cerr << "centroid shape mismatch\n"; return 3; }
    std::vector<int32_t> hier_labels(n);
    {
        std::ifstream f(hp + ".labels.u32", std::ios::binary);
        if (!f) { std::cerr << "missing labels\n"; return 4; }
        f.read(reinterpret_cast<char*>(hier_labels.data()), n * 4);
    }
    const double hier_s = TrainSeconds(hp + ".log");
    const int nthr = omp_get_max_threads();
    std::fprintf(stderr, "certrefine n=%zu d=%zu c=%zu K=%zu P=%zu B=%zu rule=%s update=%s cert=%d init=%s seed=%llu threads=%d hier_s=%.3f\n",
                 n, d, c, K, P, B, lloyd ? "lloyd" : "hartigan", batch ? "batch" : "seq", (int)use_cert, init.c_str(),
                 (unsigned long long)seed, nthr, hier_s);

    // ---- candidate graph on the hierarchical centroids (shared by the start assignment and all passes)
    auto t0 = Clock::now();
    std::vector<int32_t> nb = BuildGraph(M0.data(), c, d, K);
    const double graph0_s = secs(t0);

    // ---- start assignment
    std::vector<int32_t> labels(n);
    t0 = Clock::now();
    if (init == "exact") {
        ExactAssign(X.data(), n, d, M0.data(), c, labels.data());
    } else if (init == "snap") {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const int32_t p = hier_labels[i];
            const float* x = X.data() + i * d;
            float bs = L2sq(x, M0.data() + (size_t)p * d, d);
            int32_t best = p;
            for (size_t ci = 0; ci < K; ++ci) {
                const int32_t k = nb[(size_t)p * K + ci];
                const float s = L2sq(x, M0.data() + (size_t)k * d, d);
                if (s < bs) { bs = s; best = k; }
            }
            labels[i] = best;
        }
    } else {
        labels = hier_labels;
    }
    const double init_s = secs(t0);
    size_t init_moved = 0;
    for (size_t i = 0; i < n; ++i) init_moved += labels[i] != hier_labels[i];

    // ---- state: double sums, counts, float means
    t0 = Clock::now();
    std::vector<double> A(c * d, 0.0);
    std::vector<int64_t> cnt(c, 0);
    {
        std::vector<size_t> start(c + 1, 0), pos(c), order(n);
        for (size_t i = 0; i < n; ++i) ++cnt[labels[i]];
        for (size_t k = 0; k < c; ++k) start[k + 1] = start[k] + cnt[k];
        for (size_t k = 0; k < c; ++k) pos[k] = start[k];
        for (size_t i = 0; i < n; ++i) order[pos[labels[i]]++] = i;
#pragma omp parallel for schedule(dynamic, 16)
        for (size_t k = 0; k < c; ++k)
            for (size_t t = start[k]; t < start[k + 1]; ++t) {
                const float* x = X.data() + order[t] * d;
                double* a = A.data() + k * d;
                for (size_t j = 0; j < d; ++j) a[j] += x[j];
            }
    }
    std::vector<float> M(c * d), Mcp(c * d);
    for (size_t k = 0; k < c; ++k)
        for (size_t j = 0; j < d; ++j) M[k * d + j] = (float)(A[k * d + j] / std::max<int64_t>(cnt[k], 1));
    std::vector<float> lbk(n * K, 0.f);       // per-candidate bounds (all 0: valid but certify nothing)
    std::vector<uint8_t> lbv(n, 0);           // 1 if the bounds of point i are valid for its current slots
    std::vector<double> Lam(c, 0.0);          // net displacement of each centroid from the pass checkpoint
    const double setup_s = secs(t0);

    double total = hier_s + graph0_s + init_s + setup_s;
    std::string hist = "[{\"epoch\":0,\"total_s\":" + std::to_string(total) + ",\"init_moved\":" + std::to_string(init_moved) +
                       ",\"labels_hash\":\"" + std::to_string(Fnv1a(labels.data(), n)) + "\"}";
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> order(n);
    std::vector<int32_t> tgt(n);
    std::vector<uint64_t> th_pairs(nthr), th_skip(nthr);
    std::vector<double> th_dims(nthr);

    auto recompute = [&](int32_t k) {
        double s = 0;
        const double ik = 1.0 / (double)std::max<int64_t>(cnt[k], 1);
        for (size_t j = 0; j < d; ++j) {
            M[(size_t)k * d + j] = (float)(A[(size_t)k * d + j] * ik);
            const double u = (double)M[(size_t)k * d + j] - (double)Mcp[(size_t)k * d + j];
            s += u * u;
        }
        Lam[k] = std::sqrt(s);
    };

    for (size_t ep = 1; ep <= P; ++ep) {
        auto te = Clock::now();
        // bound transfer to the new checkpoint (pass 1: displacements are 0)
        if (use_cert && ep > 1) {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                if (!lbv[i]) continue;
                const int32_t* nbp = nb.data() + (size_t)labels[i] * K;
                float* l = lbk.data() + i * K;
                for (size_t ci = 0; ci < K; ++ci) {
                    const double v = (double)l[ci] - Lam[nbp[ci]];
                    l[ci] = (float)(v - MARGIN * std::fabs(v));
                }
            }
        }
        std::fill(Lam.begin(), Lam.end(), 0.0);
        Mcp = M;
        std::iota(order.begin(), order.end(), 0u);
        std::shuffle(order.begin(), order.end(), rng);
        const double prep_s = secs(te);
        std::fill(th_pairs.begin(), th_pairs.end(), 0);
        std::fill(th_skip.begin(), th_skip.end(), 0);
        std::fill(th_dims.begin(), th_dims.end(), 0.0);
        size_t moves = 0, rejected = 0;
        std::vector<uint8_t> touched;
        if (batch) touched.assign(c, 0);
        auto td = Clock::now();
        for (size_t b0 = 0; b0 < n; b0 += B) {
            const size_t b1 = std::min(n, b0 + B);
            // ---- parallel decide against the state at the start of the batch (whole pass for batch updates)
#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                uint64_t pairs = 0, skip = 0;
                double dims = 0;
#pragma omp for schedule(dynamic, 64)
                for (size_t t = b0; t < b1; ++t) {
                    const uint32_t i = order[t];
                    const int32_t p = labels[i];
                    const int64_t bp = cnt[p];
                    int32_t best = p;
                    if (bp > 1) {
                        const float* x = X.data() + (size_t)i * d;
                        const bool certok = use_cert && lbv[i];
                        const float sp = L2sq(x, M.data() + (size_t)p * d, d);
                        dims += d;
                        double bestw = (lloyd ? 1.0 : (double)bp / (double)(bp - 1)) * (double)sp;
                        const int32_t* nbp = nb.data() + (size_t)p * K;
                        float* l = lbk.data() + (size_t)i * K;
                        for (size_t ci = 0; ci < K; ++ci) {
                            const int32_t k = nbp[ci];
                            const int64_t bk = cnt[k];
                            const double wk = lloyd ? 1.0 : (double)bk / (double)(bk + 1);
                            ++pairs;
                            if (certok) {
                                const double lo = (double)l[ci] - Lam[k];
                                if (lo > 0.0 && wk * lo * lo >= bestw) { ++skip; continue; }
                            }
                            const float s = L2sq(x, M.data() + (size_t)k * d, d);
                            dims += d;
                            const double v = std::sqrt((double)s) - Lam[k];
                            l[ci] = (float)(v - MARGIN * std::fabs(v));
                            const double wd = wk * (double)s;
                            if (wd < bestw * (1.0 - TIE)) { bestw = wd; best = k; }
                        }
                        lbv[i] = 1;
                    }
                    tgt[t] = best;
                }
                th_pairs[tid] += pairs;
                th_skip[tid] += skip;
                th_dims[tid] += dims;
            }
            if (batch) continue;
            // ---- serial commit with validation against the current state (monotone descent)
            for (size_t t = b0; t < b1; ++t) {
                const uint32_t i = order[t];
                const int32_t q = tgt[t], p = labels[i];
                if (q == p) continue;
                if (cnt[p] <= 1) { ++rejected; continue; }
                const float* x = X.data() + (size_t)i * d;
                const double sp = L2sq(x, M.data() + (size_t)p * d, d), sq = L2sq(x, M.data() + (size_t)q * d, d);
                const double keyp = lloyd ? sp : (double)cnt[p] / (double)(cnt[p] - 1) * sp;
                const double keyq = lloyd ? sq : (double)cnt[q] / (double)(cnt[q] + 1) * sq;
                if (!(keyq < keyp * (1.0 - TIE))) { ++rejected; continue; }
                for (size_t j = 0; j < d; ++j) { A[(size_t)p * d + j] -= x[j]; A[(size_t)q * d + j] += x[j]; }
                --cnt[p];
                ++cnt[q];
                recompute(p);
                recompute(q);
                labels[i] = q;
                lbv[i] = 0;
                ++moves;
            }
        }
        if (batch) {  // apply all moves decided against the pass-start state, never emptying a cluster
            for (size_t t = 0; t < n; ++t) {
                const uint32_t i = order[t];
                const int32_t q = tgt[t], p = labels[i];
                if (q == p || cnt[p] <= 1) continue;
                const float* x = X.data() + (size_t)i * d;
                for (size_t j = 0; j < d; ++j) { A[(size_t)p * d + j] -= x[j]; A[(size_t)q * d + j] += x[j]; }
                --cnt[p];
                ++cnt[q];
                labels[i] = q;
                lbv[i] = 0;
                touched[p] = touched[q] = 1;
                ++moves;
            }
#pragma omp parallel for schedule(static)
            for (size_t k = 0; k < c; ++k)
                if (touched[k]) recompute((int32_t)k);
        }
        const double pass_s = secs(td);
        const double ep_s = secs(te);
        total += ep_s;
        uint64_t pairs = 0, skip = 0;
        double dims = 0;
        for (int t = 0; t < nthr; ++t) { pairs += th_pairs[t]; skip += th_skip[t]; dims += th_dims[t]; }
        size_t empty = 0;
        for (size_t k = 0; k < c; ++k) empty += cnt[k] == 0;
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      ",{\"epoch\":%zu,\"moves\":%zu,\"rejected\":%zu,\"pairs_total\":%llu,\"pairs_skipped\":%llu,\"skipped_pct\":%.6f,"
                      "\"dims_per_point\":%.4f,\"prep_s\":%.6f,\"pass_s\":%.6f,\"epoch_s\":%.6f,\"total_s\":%.6f,\"empty\":%zu,\"labels_hash\":\"%llu\"}",
                      ep, moves, rejected, (unsigned long long)pairs, (unsigned long long)skip, pairs ? 100.0 * skip / pairs : 0.0,
                      dims / n, prep_s, pass_s, ep_s, total, empty, (unsigned long long)Fnv1a(labels.data(), n));
        hist += buf;
        std::fprintf(stderr, "ep %zu moves=%zu rejected=%zu skipped=%.2f%% pass=%.3fs total=%.3fs\n", ep, moves, rejected,
                     pairs ? 100.0 * skip / pairs : 0.0, pass_s, total);
        if (save == "all" || (save == "ends" && (ep == 1 || ep == P)))
            WriteFvecs(outp + "_ep" + std::to_string(ep) + ".centroids.fvecs", M.data(), c, d);
    }
    hist += "]";
    std::ofstream js(outp + ".refine.json");
    js << "{\"impl\":\"cpp\",\"n\":" << n << ",\"d\":" << d << ",\"c\":" << c << ",\"K\":" << K << ",\"T\":" << B
       << ",\"rule\":\"" << (lloyd ? "lloyd" : "hartigan") << "\",\"update\":\"" << (batch ? "batch" : "seq") << "\",\"cert\":"
       << (use_cert ? "true" : "false") << ",\"init\":\"" << init << "\",\"seed\":" << seed << ",\"threads\":" << nthr
       << ",\"timing\":{\"hier_s\":" << hier_s << ",\"graph0_s\":" << graph0_s << ",\"init_assign_s\":" << init_s
       << ",\"setup_s\":" << setup_s << "},\"hist\":" << hist << "}\n";
    return 0;
}
