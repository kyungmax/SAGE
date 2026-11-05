// Uninstrumented full-batch benchmark. The default input/timing contract mirrors
// baselines/darth/hnsw-test/full_run.cpp, including a second normalization of
// prepared IP queries. This runner reports model initialization separately and
// also reports initialization + the first existing batch as a cold model cost.
// The first batch remains the requested warmup (no extra hidden search pass).
// Steady searches still include native thread/workspace setup and teardown,
// input/result allocation, optional normalization, and every DARTH prediction.
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DeclarativeRecall.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#ifdef DARTH1141_NEW
#include <darth1141/search.h>
#include <faiss/impl/VisitedTable.h>
#endif
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

template <class T>
struct Vectors {
    size_t rows, dim;
    std::vector<T> values;
};

template <class T>
static Vectors<T> read_vectors(const std::string& name, size_t rows) {
    static_assert(sizeof(T) == 4, "vecs elements are 4 bytes");
    std::ifstream in(name, std::ios::binary);
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), 4);
    if (!in || dim <= 0) {
        throw std::runtime_error("Invalid vecs header: " + name);
    }
    in.seekg(0, std::ios::end);
    const auto bytes = in.tellg();
    const size_t row_bytes = (size_t(dim) + 1) * 4;
    if (bytes < 0 || size_t(bytes) != rows * row_bytes) {
        throw std::runtime_error("Unexpected vecs size: " + name);
    }
    in.seekg(0);
    Vectors<T> result{rows, size_t(dim), std::vector<T>(rows * dim)};
    for (size_t q = 0; q < rows; ++q) {
        int32_t current;
        in.read(reinterpret_cast<char*>(&current), 4);
        in.read(reinterpret_cast<char*>(result.values.data() + q * dim), dim * 4);
        if (!in || current != dim) {
            throw std::runtime_error("Invalid vecs row: " + name);
        }
    }
    return result;
}

template <class T>
static void write_binary(const std::string& name, const std::vector<T>& values) {
    std::ofstream out(name, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    if (!out) {
        throw std::runtime_error("Cannot write " + name);
    }
}

template <class T>
static std::string hash_bytes(const std::vector<T>& values) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const auto* data = reinterpret_cast<const unsigned char*>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(T); ++i) {
        hash = (hash ^ data[i]) * UINT64_C(1099511628211);
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

int main(int argc, char** argv) {
    try {
        static_assert(sizeof(faiss::idx_t) == 8, "Output IDs must be int64");
        const std::set<std::string> allowed{
                "--index", "--query-file", "--gt-file", "--gt-dist-file",
                "--queries", "--k", "--threads", "--ef", "--initial-interval",
                "--min-interval", "--target", "--model", "--warmups",
                "--repeats", "--result-prefix", "--mode", "--visited", "--normalize",
                "--audit-prefix", "--model-lifetime"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || !allowed.count(argv[i]) ||
                !args.emplace(argv[i], argv[i + 1]).second) {
                throw std::runtime_error("Expected unique supported --name value pairs");
            }
        }
        const auto required = [&](const char* key) {
            if (!args.count(key)) {
                throw std::runtime_error(std::string("Missing ") + key);
            }
            return args.at(key);
        };
        const auto integer = [&](const char* key, bool zero = false) -> size_t {
            size_t consumed = 0;
            const auto text = required(key);
            const auto value = std::stoll(text, &consumed);
            if (consumed != text.size() || value < (zero ? 0 : 1) ||
                value > std::numeric_limits<int>::max()) {
                throw std::runtime_error(std::string("Invalid integer ") + key);
            }
            return size_t(value);
        };
        const size_t nq = integer("--queries"), k = integer("--k"),
                     threads = integer("--threads"), ef = integer("--ef"),
                     initial = integer("--initial-interval"), minimum = integer("--min-interval"),
                     warmups = integer("--warmups", true), repeats = integer("--repeats");
        size_t target_consumed = 0;
        const auto target_text = required("--target");
        const double target = std::stod(target_text, &target_consumed);
        if (target_consumed != target_text.size() || !(target > 0 && target <= 1) ||
            minimum > initial || ef < k) {
            throw std::runtime_error("Invalid target/interval/ef");
        }
        const std::string mode = args.count("--mode") ? args.at("--mode") : "darth";
        const std::string visited = args.count("--visited") ? args.at("--visited") : "auto";
        const std::string normalize = args.count("--normalize") ? args.at("--normalize") : "1";
        const std::string model_lifetime = args.count("--model-lifetime") ? args.at("--model-lifetime") : "persistent";
        if (mode != "darth" && mode != "vanilla" && mode != "observe") {
            throw std::runtime_error("--mode must be darth, vanilla, or observe");
        }
        if (visited != "auto" && visited != "dense" && visited != "hash") {
            throw std::runtime_error("--visited must be auto, dense, or hash");
        }
        if (normalize != "0" && normalize != "1") {
            throw std::runtime_error("--normalize must be 0 or 1");
        }
        if (model_lifetime != "persistent" && model_lifetime != "per-call") {
            throw std::runtime_error("--model-lifetime must be persistent or per-call");
        }
        if (args.count("--audit-prefix") && mode != "darth") {
            throw std::runtime_error("--audit-prefix requires --mode darth");
        }
#ifndef DARTH1141_NEW
        if (visited == "hash" || mode == "observe") {
            throw std::runtime_error("Frozen old runner supports dense visited and darth/vanilla only");
        }
#endif
        const auto prefix = required("--result-prefix");
        omp_set_dynamic(0);
        omp_set_num_threads(int(threads));
        const auto input_start = Clock::now();
        auto raw = read_vectors<float>(required("--query-file"), nq);
        auto truth = read_vectors<int32_t>(required("--gt-file"), nq);
        auto truth_dist = read_vectors<float>(required("--gt-dist-file"), nq);
        if (truth.dim != k || truth_dist.dim != k) {
            throw std::runtime_error("GT must contain exactly k columns");
        }
        std::vector<faiss::idx_t> gt(truth.values.begin(), truth.values.end());
        const auto input_end = Clock::now();
        const auto index_start = Clock::now();
        std::unique_ptr<faiss::Index> owner(faiss::read_index(required("--index").c_str()));
        const auto index_end = Clock::now();
        auto* index = dynamic_cast<faiss::IndexHNSWFlat*>(owner.get());
        if (!index || index->d != int(raw.dim) || index->ntotal < faiss::idx_t(k) ||
            (index->metric_type != faiss::METRIC_L2 &&
             index->metric_type != faiss::METRIC_INNER_PRODUCT)) {
            throw std::runtime_error("Expected matching L2/IP HNSWFlat index");
        }
        index->hnsw.efSearch = int(ef);
        std::string effective_visited = "dense";
#ifdef DARTH1141_NEW
        if (visited == "dense") {
            index->hnsw.use_visited_hashset = false;
        } else if (visited == "hash") {
            index->hnsw.use_visited_hashset = true;
        } else {
            index->hnsw.use_visited_hashset.reset();
        }
        effective_visited = index->hnsw.use_visited_hashset.value_or(
                size_t(index->ntotal) >= faiss::visited_table_hashset_threshold) ? "hash" : "dense";
#endif
        const bool renormalize = normalize == "1" &&
                                 index->metric_type == faiss::METRIC_INNER_PRODUCT;
        const auto base_model_start = Clock::now();
        std::unique_ptr<faiss::DARTHPredictorHNSW> predictor;
        if (mode != "vanilla") {
            faiss::DeclarativeRecallDataManager empty;
            const auto model = required("--model");
            predictor = std::make_unique<faiss::DARTHPredictorHNSW>(
                    empty, target, int(initial), int(minimum), false, model.c_str());
        }
        const auto base_model_end = Clock::now();
        const auto pool_start = Clock::now();
#ifdef DARTH1141_NEW
        std::unique_ptr<darth1141::PredictorPool> predictor_pool;
        if (predictor && model_lifetime == "persistent") {
            predictor_pool = std::make_unique<darth1141::PredictorPool>(
                    *predictor, int(std::min(nq, threads)));
        }
        const size_t pool_capacity = predictor_pool ? size_t(predictor_pool->capacity()) : 0;
#else
        const size_t pool_capacity = 0;
#endif
        const auto pool_end = Clock::now();
        const double base_model_seconds = predictor ? seconds(base_model_start, base_model_end) : 0;
        const double pool_seconds = pool_capacity ? seconds(pool_start, pool_end) : 0;
        const double total_model_seconds = base_model_seconds + pool_seconds;
        std::vector<faiss::idx_t> reference_labels;
        std::vector<float> reference_distances;
        size_t reference_ndis = 0;
        std::cout << std::setprecision(17);
        std::cout << "{\"phase\":\"model_initialization\",\"model_lifetime\":\"" << model_lifetime
                  << "\",\"input_load_seconds\":" << seconds(input_start, input_end)
                  << ",\"index_load_seconds\":" << seconds(index_start, index_end)
                  << ",\"base_model_load_seconds\":" << base_model_seconds
                  << ",\"predictor_pool_load_seconds\":" << pool_seconds
                  << ",\"total_model_init_seconds\":" << total_model_seconds
                  << ",\"pool_capacity\":" << pool_capacity
                  << ",\"worker_models_loaded_initially\":" << (predictor ? std::max<size_t>(1, pool_capacity) : 0)
                  << ",\"pool_initialization_includes_model_identity_verification\":"
                  << (pool_capacity ? "true" : "false")
                  << "}\n" << std::flush;
        for (size_t run = 0; run < warmups + repeats; ++run) {
            faiss::hnsw_stats.reset();
            const auto start = Clock::now();
            std::vector<float> queries(raw.values);
            if (renormalize) {
                faiss::fvec_renorm_L2(raw.dim, nq, queries.data());
            }
            std::vector<faiss::idx_t> labels(nq * k, -1);
            std::vector<float> distances(nq * k);
            if (predictor) {
                predictor->data_manager = faiss::DeclarativeRecallDataManager(
                        distances.data(), labels.data(), gt.data(), truth_dist.values.data(),
                        nq, raw.dim, k, queries.data(), nullptr, nullptr, index->ntotal, gt.data(), k);
            }
            const auto api_start = Clock::now();
            if (mode == "vanilla") {
                index->search(nq, queries.data(), k, distances.data(), labels.data());
            } else {
#ifdef DARTH1141_NEW
                darth1141::SearchOptions options;
                options.mode = mode == "observe" ? darth1141::Mode::ObserveOnly : darth1141::Mode::Darth;
                options.threads = int(threads);
                // No trace wrapper, model interposition, logging, or diagnostics.
                options.diagnostics = nullptr;
                options.collect_trace = false;
                options.predictor_pool = predictor_pool.get();
                darth1141::search(*index, nq, queries.data(), k, distances.data(), labels.data(),
                                 predictor.get(), nullptr, options);
#else
                index->search_DARTH(nq, queries.data(), k, distances.data(), labels.data(), *predictor);
#endif
            }
            const auto end = Clock::now();
            size_t hits = 0;
            for (size_t q = 0; q < nq; ++q) {
                for (size_t j = 0; j < k; ++j) {
                    const auto id = labels[q * k + j];
                    if (id < 0 || id >= index->ntotal || !std::isfinite(distances[q * k + j])) {
                        throw std::runtime_error("Invalid search result");
                    }
                    for (size_t p = 0; p < j; ++p) {
                        if (labels[q * k + p] == id) {
                            throw std::runtime_error("Duplicate result ID");
                        }
                    }
                    hits += std::find(gt.begin() + q * k, gt.begin() + (q + 1) * k, id) !=
                            gt.begin() + (q + 1) * k;
                }
            }
            if (run == 0) {
                reference_labels = labels;
                reference_distances = distances;
                reference_ndis = faiss::hnsw_stats.ndis;
            } else if (labels != reference_labels ||
                       std::memcmp(distances.data(), reference_distances.data(), distances.size() * sizeof(float)) ||
                       reference_ndis != faiss::hnsw_stats.ndis) {
                throw std::runtime_error("Warmup/repeat results or work differ");
            }
            std::cout << "{\"implementation\":\""
#ifdef DARTH1141_NEW
                      << "darth_faiss1141"
#else
                      << "darth_frozen18"
#endif
                      << "\",\"mode\":\"" << mode
                      << "\",\"model_lifetime\":\"" << model_lifetime
                      << "\",\"visited_requested\":\"" << visited
                      << "\",\"visited_effective\":\"" << effective_visited
                      << "\",\"renormalize_ip\":" << (renormalize ? "true" : "false")
                      << ",\"query_count\":" << nq << ",\"k\":" << k
                      << ",\"threads\":" << threads << ",\"ef\":" << ef
                      << ",\"phase\":\"" << (run < warmups ? "warmup" : "timed")
                      << "\",\"repeat\":" << (run < warmups ? run + 1 : run - warmups + 1)
                      << ",\"target_recall\":" << target << ",\"hits\":" << hits
                      << ",\"recall\":" << double(hits) / (nq * k)
                      << ",\"ndis\":" << faiss::hnsw_stats.ndis
                      << ",\"e2e_seconds\":" << seconds(start, end)
                      << ",\"search_api_seconds\":" << seconds(api_start, end)
                      << ",\"is_first_search\":" << (run == 0 ? "true" : "false");
            if (run == 0) {
                // This explicitly counts one-time model loading once; it is
                // not subtracted from an observed search time. Index/query
                // file loading is separately reported above, not included.
                std::cout << ",\"cold_combined_model_init_plus_first_search_seconds\":"
                          << total_model_seconds + seconds(start, end);
            }
            std::cout
                      << ",\"ids_fnv1a64\":\"" << hash_bytes(labels)
                      << "\",\"distances_fnv1a64\":\"" << hash_bytes(distances)
                      << "\"}\n" << std::flush;
        }
        write_binary(prefix + ".ids.i64", reference_labels);
        write_binary(prefix + ".distances.f32", reference_distances);
        if (args.count("--audit-prefix")) {
            // A separate, explicitly untimed pass provides stopping counters.
            // It must reproduce the uninstrumented timed passes exactly.
            const auto audit_prefix = args.at("--audit-prefix");
            std::vector<float> queries(raw.values);
            if (renormalize) {
                faiss::fvec_renorm_L2(raw.dim, nq, queries.data());
            }
            std::vector<faiss::idx_t> labels(nq * k, -1);
            std::vector<float> distances(nq * k);
            auto log_filename = audit_prefix + ".final.csv";
            predictor->data_manager = faiss::DeclarativeRecallDataManager(
                    distances.data(), labels.data(), gt.data(), truth_dist.values.data(),
                    nq, raw.dim, k, queries.data(), log_filename.data(), nullptr, index->ntotal, gt.data(), k);
            predictor->init_log_file();
            faiss::hnsw_stats.reset();
#ifdef DARTH1141_NEW
            std::vector<darth1141::QueryDiagnostics> diagnostics;
            darth1141::SearchOptions audit_options;
            audit_options.mode = darth1141::Mode::Darth;
            audit_options.threads = int(threads);
            audit_options.diagnostics = &diagnostics;
            audit_options.collect_trace = false;
            audit_options.predictor_pool = predictor_pool.get();
            darth1141::search(*index, nq, queries.data(), k, distances.data(), labels.data(),
                             predictor.get(), nullptr, audit_options);
#else
            index->search_DARTH(nq, queries.data(), k, distances.data(), labels.data(), *predictor);
#endif
            predictor->close_log_file();
            if (labels != reference_labels ||
                std::memcmp(distances.data(), reference_distances.data(), distances.size() * sizeof(float)) ||
                reference_ndis != faiss::hnsw_stats.ndis) {
                throw std::runtime_error("Untimed audit differs from uninstrumented timed runs");
            }
            std::cout << "{\"phase\":\"untimed_audit\",\"query_count\":" << nq
                      << ",\"matches_timed_outputs\":true,\"ndis\":" << faiss::hnsw_stats.ndis;
#ifdef DARTH1141_NEW
            std::ofstream counters(audit_prefix + ".counters.csv");
            counters << "qid,actual_distance_evaluations,upper_distance_evaluations,logical_ndis,nstep,insertions,"
                        "predictor_calls,final_interval,first_nn_distance,predicted_recall,early_terminated,uses_hashset,completed_expansions,started_expansions,unfinished_expansions\n";
            counters << std::setprecision(17);
            size_t early_terminated = 0;
            uint64_t total_calls = 0, actual_distances = 0, logical_distances = 0;
            for (size_t q = 0; q < diagnostics.size(); ++q) {
                const auto& info = diagnostics[q];
                early_terminated += info.early_terminated;
                total_calls += info.predictor_calls;
                actual_distances += info.actual_distance_evaluations;
                logical_distances += info.logical_ndis;
                counters << q << ',' << info.actual_distance_evaluations << ','
                         << info.upper_distance_evaluations << ',' << info.logical_ndis << ','
                         << info.nstep << ',' << info.insertions << ',' << info.predictor_calls << ','
                         << info.final_interval << ',' << info.first_nn_distance << ','
                         << info.predicted_recall << ',' << info.early_terminated << ','
                         << info.uses_hashset << ',' << info.completed_expansions << ','
                         << info.started_expansions << ',' << info.unfinished_expansions << '\n';
            }
            if (!counters) {
                throw std::runtime_error("Cannot write audit counters");
            }
            std::cout << ",\"early_terminated_queries\":" << early_terminated
                      << ",\"predictor_calls\":" << total_calls
                      << ",\"actual_distance_evaluations\":" << actual_distances
                      << ",\"logical_ndis\":" << logical_distances;
#endif
            std::cout << "}\n" << std::flush;
        }
        // The pool contains only model handles; all batch-local buffers have
        // already been destroyed. Report final ownership release separately.
        const auto model_free_start = Clock::now();
#ifdef DARTH1141_NEW
        predictor_pool.reset();
#endif
        predictor.reset();
        const auto model_free_end = Clock::now();
        std::cout << "{\"phase\":\"model_shutdown\",\"model_lifetime\":\"" << model_lifetime
                  << "\",\"model_free_seconds\":" << seconds(model_free_start, model_free_end)
                  << "}\n" << std::flush;
    } catch (const std::exception& error) {
        std::cerr << "DARTH full-run benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
