#include <H5Cpp.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hnswlib/adaptive_ef.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Config {
    std::string phase;
    std::string dataset;
    std::string metric = "auto";
    fs::path data_path;
    fs::path dataset_root;
    fs::path experiments_root = "./experiments";
    fs::path index_root;
    fs::path output_json;
    fs::path per_query_csv;
    int m = 16;
    int ef_construction = 500;
    int k = 10;
    int num_threads = std::max(1u, std::thread::hardware_concurrency() / 4);
    int warmup_runs = 1;
    int measured_runs = 5;
    int ef_upper_bound = 5000;
    int sample_size = 200;
    float expected_recall = 0.95f;
    float quantile_step = 0.001f;
    size_t statics_length = 0;
    bool statics_length_overridden = false;
    bool force = false;
    bool parallel_queries = false;
};

struct ArtifactPaths {
    fs::path dataset_path;
    fs::path dataset_root;
    fs::path experiments_root;
    fs::path index_path;
    fs::path estimator_path;
    fs::path samplings_path;
    fs::path ef_adaptor_path;
};

struct LoadedIndex {
    std::shared_ptr<hnswlib::SpaceInterface<float>> space;
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> index;
};

struct BatchResult {
    std::vector<std::vector<size_t>> labels;
    std::vector<double> query_latency_ms;
    double batch_latency_ms = 0.0;
};

struct OnlineStats {
    int query_count = 0;
    double achieved_recall = 0.0;
    double mean_batch_latency_ms = 0.0;
    double p50_batch_latency_ms = 0.0;
    double p95_batch_latency_ms = 0.0;
    double mean_query_latency_ms = 0.0;
    double p50_query_latency_ms = 0.0;
    double p95_query_latency_ms = 0.0;
    double p99_query_latency_ms = 0.0;
    double qps = 0.0;
};

std::string usage() {
    return R"(Usage:
  backend_runner --phase build|offline|online --dataset <dataset-stem> [options]

Required:
  --phase <build|offline|online>
  --dataset <name-or-file>

Common options:
  --experiments-root <dir>       Artifact root. Default: ./experiments
  --dataset-root <dir>           HDF5 root. Default: <experiments-root>/data
  --data-path <file.hdf5>        Explicit dataset file
  --index-root <dir>             HNSW index root. Default: <experiments-root>/index
  --output-json <path>           Write machine-readable result JSON
  --metric <auto|cd|ipd|l2>      Ada-EF supports cd/ipd for offline/online. Default: auto
  --m <int>                      HNSW M. Default: 16
  --ef-construction <int>        HNSW efConstruction. Default: 500
  --k <int>                      Recall@k. Default: 10
  --num-threads <int>            Build or query threads. Default: hardware/4
  --expected-recall <float>      Ada-EF target recall. Default: 0.95
  --quantile-step <float>        Ada-EF score quantile step. Default: 0.001
  --statics-length <int>         Ada-EF early-statistics length. Default: M-aware 2-hop
  --ef-upper-bound <int>         Ada-EF offline search cap. Default: 5000
  --sample-size <int>            Offline sample size. Default: 200
  --warmup-runs <int>            Online warmup runs. Default: 1
  --measured-runs <int>          Online measured runs. Default: 5
  --parallel-queries             Run online queries with OpenMP
  --per-query-csv <path>         Optional per-query chosen-ef CSV
  --force                        Rebuild existing artifacts
)";
}

size_t default_statics_length_for_m(int m) {
    const size_t base_degree = static_cast<size_t>(2 * m);
    return 1 + base_degree + (base_degree - 1) * base_degree;
}

bool starts_with_dash(const std::string& value) {
    return !value.empty() && value[0] == '-';
}

std::string require_value(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc || starts_with_dash(argv[i + 1])) {
        throw std::invalid_argument("Missing value for " + flag);
    }
    return argv[++i];
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << usage();
            std::exit(0);
        } else if (arg == "--phase") {
            cfg.phase = require_value(i, argc, argv, arg);
        } else if (arg == "--dataset") {
            cfg.dataset = require_value(i, argc, argv, arg);
        } else if (arg == "--metric") {
            cfg.metric = require_value(i, argc, argv, arg);
        } else if (arg == "--data-path") {
            cfg.data_path = require_value(i, argc, argv, arg);
        } else if (arg == "--dataset-root") {
            cfg.dataset_root = require_value(i, argc, argv, arg);
        } else if (arg == "--experiments-root") {
            cfg.experiments_root = require_value(i, argc, argv, arg);
        } else if (arg == "--index-root") {
            cfg.index_root = require_value(i, argc, argv, arg);
        } else if (arg == "--output-json") {
            cfg.output_json = require_value(i, argc, argv, arg);
        } else if (arg == "--per-query-csv") {
            cfg.per_query_csv = require_value(i, argc, argv, arg);
        } else if (arg == "--m") {
            cfg.m = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--ef-construction") {
            cfg.ef_construction = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--k") {
            cfg.k = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--num-threads") {
            cfg.num_threads = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--warmup-runs") {
            cfg.warmup_runs = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--measured-runs") {
            cfg.measured_runs = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--expected-recall") {
            cfg.expected_recall = std::stof(require_value(i, argc, argv, arg));
        } else if (arg == "--quantile-step") {
            cfg.quantile_step = std::stof(require_value(i, argc, argv, arg));
        } else if (arg == "--statics-length") {
            cfg.statics_length = static_cast<size_t>(std::stoull(require_value(i, argc, argv, arg)));
            cfg.statics_length_overridden = true;
        } else if (arg == "--ef-upper-bound") {
            cfg.ef_upper_bound = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--sample-size") {
            cfg.sample_size = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--force") {
            cfg.force = true;
        } else if (arg == "--parallel-queries") {
            cfg.parallel_queries = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (cfg.phase.empty()) {
        throw std::invalid_argument("--phase is required");
    }
    if (cfg.phase != "build" && cfg.phase != "offline" && cfg.phase != "online") {
        throw std::invalid_argument("--phase must be build, offline, or online");
    }
    if (cfg.dataset.empty()) {
        throw std::invalid_argument("--dataset is required");
    }
    if (cfg.k <= 0 || cfg.m <= 0 || cfg.ef_construction <= 0 || cfg.num_threads <= 0) {
        throw std::invalid_argument("--k, --m, --ef-construction, and --num-threads must be positive");
    }
    if (cfg.sample_size <= 0 || cfg.warmup_runs < 0 || cfg.measured_runs <= 0 || cfg.ef_upper_bound <= 0) {
        throw std::invalid_argument("--sample-size, --measured-runs, and --ef-upper-bound must be positive");
    }
    if (cfg.statics_length_overridden && cfg.statics_length == 0) {
        throw std::invalid_argument("--statics-length must be positive");
    }
    if (!cfg.statics_length_overridden) {
        cfg.statics_length = default_statics_length_for_m(cfg.m);
    }
    return cfg;
}

std::string dataset_stem(const std::string& dataset) {
    fs::path p(dataset);
    std::string name = p.filename().string();
    if (name.size() > 5 && name.substr(name.size() - 5) == ".hdf5") {
        name.resize(name.size() - 5);
    }
    return name;
}

std::string infer_metric(const Config& cfg) {
    if (cfg.metric != "auto") {
        return cfg.metric;
    }
    std::string lower = dataset_stem(cfg.dataset);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower.find("-ip") != std::string::npos || lower.find("inner") != std::string::npos ||
        lower.find("ipd") != std::string::npos || lower.find("msmarco") != std::string::npos) {
        return "ipd";
    }
    return "cd";
}

std::string abs_string(const fs::path& path) {
    return fs::absolute(path).lexically_normal().string();
}

void ensure_dir(const fs::path& path) {
    if (!path.empty()) {
        fs::create_directories(path);
    }
}

void ensure_parent(const fs::path& path) {
    if (!path.empty()) {
        ensure_dir(path.parent_path());
    }
}

uintmax_t file_size_or_zero(const fs::path& path) {
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        return 0;
    }
    return fs::file_size(path, ec);
}

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename T>
double mean_value(const std::vector<T>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    double pos = p * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }
    double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

std::string quote(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

void write_error_json(const fs::path& output_json, const Config& cfg, const std::string& message) {
    if (output_json.empty()) {
        return;
    }
    ensure_parent(output_json);
    std::ofstream out(output_json);
    out << "{\n"
        << "  \"backend\": \"adaef\",\n"
        << "  \"phase\": " << quote(cfg.phase) << ",\n"
        << "  \"dataset\": " << quote(dataset_stem(cfg.dataset)) << ",\n"
        << "  \"status\": \"error\",\n"
        << "  \"error\": " << quote(message) << "\n"
        << "}\n";
}

void read_float_matrix(H5::H5File& file, const std::string& name, hnswdis::MatrixXf& matrix) {
    H5::DataSet dataset = file.openDataSet(name);
    H5::DataSpace dataspace = dataset.getSpace();
    if (dataspace.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("HDF5 dataset is not rank-2: " + name);
    }
    hsize_t dims[2] = {0, 0};
    dataspace.getSimpleExtentDims(dims, nullptr);
    matrix.resize(static_cast<Eigen::Index>(dims[0]), static_cast<Eigen::Index>(dims[1]));
    dataset.read(matrix.data(), H5::PredType::NATIVE_FLOAT);
}

void read_int_matrix(H5::H5File& file, const std::string& name, hnswdis::MatrixXi& matrix) {
    H5::DataSet dataset = file.openDataSet(name);
    H5::DataSpace dataspace = dataset.getSpace();
    if (dataspace.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("HDF5 dataset is not rank-2: " + name);
    }
    hsize_t dims[2] = {0, 0};
    dataspace.getSimpleExtentDims(dims, nullptr);
    matrix.resize(static_cast<Eigen::Index>(dims[0]), static_cast<Eigen::Index>(dims[1]));
    dataset.read(matrix.data(), H5::PredType::NATIVE_INT);
}

std::shared_ptr<hnswdis::MatrixXf> load_train(const fs::path& hdf5_path) {
    auto train = std::make_shared<hnswdis::MatrixXf>();
    H5::H5File file(hdf5_path.string(), H5F_ACC_RDONLY);
    read_float_matrix(file, "train", *train);
    return train;
}

std::tuple<
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXi>>
load_full_dataset(const fs::path& hdf5_path) {
    auto queries = std::make_shared<hnswdis::MatrixXf>();
    auto train = std::make_shared<hnswdis::MatrixXf>();
    auto neighbors = std::make_shared<hnswdis::MatrixXi>();
    H5::H5File file(hdf5_path.string(), H5F_ACC_RDONLY);
    read_float_matrix(file, "train", *train);
    read_float_matrix(file, "test", *queries);
    read_int_matrix(file, "neighbors", *neighbors);
    return {queries, train, neighbors};
}

void normalize_matrix(hnswdis::MatrixXf& matrix) {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        float norm = matrix.row(i).norm();
        matrix.row(i) *= 1.0f / (norm + 1e-30f);
    }
}

ArtifactPaths build_paths(const Config& cfg) {
    const std::string stem = dataset_stem(cfg.dataset);
    ArtifactPaths paths;
    paths.experiments_root = cfg.experiments_root;
    paths.dataset_root = cfg.dataset_root.empty() ? (cfg.experiments_root / "data") : cfg.dataset_root;

    if (!cfg.data_path.empty()) {
        paths.dataset_path = cfg.data_path;
    } else {
        fs::path dataset_as_path(cfg.dataset);
        if (dataset_as_path.has_parent_path()) {
            paths.dataset_path = dataset_as_path;
        } else {
            paths.dataset_path = paths.dataset_root / (dataset_as_path.extension() == ".hdf5" ? dataset_as_path.string()
                                                                                               : (stem + ".hdf5"));
        }
    }

    fs::path index_root = cfg.index_root.empty() ? (cfg.experiments_root / "index") : cfg.index_root;
    paths.index_path = index_root / (stem + "-M" + std::to_string(cfg.m) +
                                     "-efc-" + std::to_string(cfg.ef_construction) + "-parallel.hnsw");
    paths.estimator_path = cfg.experiments_root / "statistics" /
                           (stem + "-estimator--k-" + std::to_string(cfg.k) + ".bin");
    paths.samplings_path = cfg.experiments_root / "sampling" /
                           (stem + "-samplings--k" + std::to_string(cfg.k) + "-ef.bin");
    std::string ef_name = stem + "-ef_adaptor--k" + std::to_string(cfg.k);
    if (cfg.statics_length_overridden || cfg.m != 16) {
        ef_name += "-sl" + std::to_string(cfg.statics_length);
    }
    ef_name += "-ef.bin";
    paths.ef_adaptor_path = cfg.experiments_root / "estimation_table" / ef_name;
    return paths;
}

void ensure_artifact_dirs(const ArtifactPaths& paths) {
    ensure_dir(paths.experiments_root);
    ensure_dir(paths.dataset_root);
    ensure_parent(paths.index_path);
    ensure_parent(paths.estimator_path);
    ensure_parent(paths.samplings_path);
    ensure_parent(paths.ef_adaptor_path);
}

void validate_metric_for_adaef(const std::string& metric) {
    if (metric != "cd" && metric != "ipd") {
        throw std::invalid_argument("Ada-EF offline/online supports metric=cd or metric=ipd. Got: " + metric);
    }
}

LoadedIndex load_index(const fs::path& index_path, const std::string& metric, int dim) {
    LoadedIndex loaded;
    loaded.space = hnswdis::init_space(metric, dim);
    loaded.index = std::make_shared<hnswlib::HierarchicalNSW<float>>(loaded.space.get(), index_path.string());
    return loaded;
}

template <typename PriorityQueue>
std::vector<size_t> labels_from_queue(PriorityQueue queue) {
    size_t count = queue.size();
    std::vector<size_t> labels(count);
    while (!queue.empty()) {
        labels[--count] = static_cast<size_t>(queue.top().second);
        queue.pop();
    }
    return labels;
}

double recall_one(const hnswdis::MatrixXi& ground_truth, const std::vector<size_t>& labels, int row, int k) {
    std::unordered_set<int> truth;
    truth.reserve(static_cast<size_t>(k) * 2);
    for (int j = 0; j < k; ++j) {
        truth.insert(ground_truth(row, j));
    }
    int correct = 0;
    for (size_t label : labels) {
        if (truth.find(static_cast<int>(label)) != truth.end()) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(k);
}

hnswdis::MatrixXf sample_rows(const hnswdis::MatrixXf& data, int requested_sample_size) {
    const int sample_size = std::min<int>(requested_sample_size, static_cast<int>(data.rows()));
    if (sample_size <= 0) {
        throw std::runtime_error("Cannot sample from an empty train matrix");
    }
    std::vector<int> indices(static_cast<size_t>(data.rows()));
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123456789);
    std::shuffle(indices.begin(), indices.end(), rng);

    hnswdis::MatrixXf sample(sample_size, data.cols());
    for (int i = 0; i < sample_size; ++i) {
        sample.row(i) = data.row(indices[static_cast<size_t>(i)]);
    }
    return sample;
}

void ensure_sampling_file(const Config& cfg,
                          const ArtifactPaths& paths,
                          const hnswdis::MatrixXf& data,
                          const std::string& metric,
                          double& sampling_ms,
                          bool& sampling_reused) {
    if (fs::exists(paths.samplings_path) && !cfg.force) {
        sampling_reused = true;
        sampling_ms = 0.0;
        return;
    }

    auto start = Clock::now();
    hnswdis::MatrixXf sample_queries = sample_rows(data, cfg.sample_size);
    hnswdis::MatrixXi sample_ground_truth =
        hnswdis::compute_ground_truth_batch_parallel4(sample_queries, data, metric, cfg.k);
    hnswdis::serialize_samplings(paths.samplings_path.string(), sample_queries, sample_ground_truth);
    sampling_ms = elapsed_ms(start, Clock::now());
    sampling_reused = false;
}

std::shared_ptr<hnswdis::Estimator> ensure_estimator_file(const Config& cfg,
                                                          const ArtifactPaths& paths,
                                                          const hnswdis::MatrixXf& data,
                                                          const std::string& metric,
                                                          double& estimator_ms,
                                                          bool& estimator_reused) {
    if (fs::exists(paths.estimator_path) && !cfg.force) {
        estimator_reused = true;
        estimator_ms = 0.0;
        return hnswdis::load_estimator_from_file(paths.estimator_path.string());
    }

    auto start = Clock::now();
    std::shared_ptr<hnswdis::Estimator> estimator = hnswdis::init_estimator(metric, data);
    hnswdis::save_estimator_to_file(*estimator, paths.estimator_path.string());
    estimator_ms = elapsed_ms(start, Clock::now());
    estimator_reused = false;
    return estimator;
}

void write_artifacts_json(std::ostream& out, const ArtifactPaths& paths) {
    out << "  \"artifacts\": {\n"
        << "    \"ef_adaptor_path\": " << quote(abs_string(paths.ef_adaptor_path)) << ",\n"
        << "    \"samplings_path\": " << quote(abs_string(paths.samplings_path)) << ",\n"
        << "    \"estimator_path\": " << quote(abs_string(paths.estimator_path)) << ",\n"
        << "    \"index_path\": " << quote(abs_string(paths.index_path)) << ",\n"
        << "    \"dataset_path\": " << quote(abs_string(paths.dataset_path)) << ",\n"
        << "    \"dataset_root\": " << quote(abs_string(paths.dataset_root)) << ",\n"
        << "    \"experiments_root\": " << quote(abs_string(paths.experiments_root)) << "\n"
        << "  }\n";
}

void write_build_json(const Config& cfg,
                      const ArtifactPaths& paths,
                      const std::string& metric,
                      int base_count,
                      int dim,
                      double index_build_ms,
                      bool reused_index) {
    if (cfg.output_json.empty()) {
        return;
    }
    ensure_parent(cfg.output_json);
    std::ofstream out(cfg.output_json);
    out << std::setprecision(10)
        << "{\n"
        << "  \"backend\": \"adaef\",\n"
        << "  \"phase\": \"build\",\n"
        << "  \"dataset\": " << quote(dataset_stem(cfg.dataset)) << ",\n"
        << "  \"status\": \"ok\",\n"
        << "  \"metric\": " << quote(metric) << ",\n"
        << "  \"metrics\": {\n"
        << "    \"base_count\": " << base_count << ",\n"
        << "    \"dim\": " << dim << ",\n"
        << "    \"index_build_ms\": " << index_build_ms << ",\n"
        << "    \"index_bytes\": " << file_size_or_zero(paths.index_path) << ",\n"
        << "    \"num_threads\": " << cfg.num_threads << ",\n"
        << "    \"reused_index\": " << (reused_index ? 1 : 0) << "\n"
        << "  },\n";
    write_artifacts_json(out, paths);
    out << "}\n";
}

void write_offline_json(const Config& cfg,
                        const ArtifactPaths& paths,
                        const std::string& metric,
                        double estimator_ms,
                        double sampling_ms,
                        double ef_adaptor_ms,
                        bool estimator_reused,
                        bool sampling_reused,
                        bool ef_adaptor_reused,
                        float weighted_average_ef) {
    if (cfg.output_json.empty()) {
        return;
    }
    ensure_parent(cfg.output_json);
    const double extra_offline_ms = estimator_ms + sampling_ms + ef_adaptor_ms;
    std::ofstream out(cfg.output_json);
    out << std::setprecision(10)
        << "{\n"
        << "  \"backend\": \"adaef\",\n"
        << "  \"phase\": \"offline\",\n"
        << "  \"dataset\": " << quote(dataset_stem(cfg.dataset)) << ",\n"
        << "  \"status\": \"ok\",\n"
        << "  \"metric\": " << quote(metric) << ",\n"
        << "  \"metrics\": {\n"
        << "    \"estimator_ms\": " << estimator_ms << ",\n"
        << "    \"sampling_ms\": " << sampling_ms << ",\n"
        << "    \"ef_adaptor_ms\": " << ef_adaptor_ms << ",\n"
        << "    \"extra_offline_ms\": " << extra_offline_ms << ",\n"
        << "    \"total_offline_ms\": " << extra_offline_ms << ",\n"
        << "    \"estimator_bytes\": " << file_size_or_zero(paths.estimator_path) << ",\n"
        << "    \"samplings_bytes\": " << file_size_or_zero(paths.samplings_path) << ",\n"
        << "    \"ef_adaptor_bytes\": " << file_size_or_zero(paths.ef_adaptor_path) << ",\n"
        << "    \"weighted_average_ef\": " << weighted_average_ef << ",\n"
        << "    \"expected_recall\": " << cfg.expected_recall << ",\n"
        << "    \"statics_length\": " << cfg.statics_length << ",\n"
        << "    \"statics_length_override\": " << (cfg.statics_length_overridden ? 1 : 0) << ",\n"
        << "    \"statics_length_auto\": " << (cfg.statics_length_overridden ? 0 : 1) << ",\n"
        << "    \"num_threads\": " << cfg.num_threads << ",\n"
        << "    \"reused_estimator\": " << (estimator_reused ? 1 : 0) << ",\n"
        << "    \"reused_sampling\": " << (sampling_reused ? 1 : 0) << ",\n"
        << "    \"reused_ef_adaptor\": " << (ef_adaptor_reused ? 1 : 0) << "\n"
        << "  },\n";
    write_artifacts_json(out, paths);
    out << "}\n";
}

void write_online_json(const Config& cfg,
                       const ArtifactPaths& paths,
                       const std::string& metric,
                       const OnlineStats& stats,
                       float weighted_average_ef) {
    if (cfg.output_json.empty()) {
        return;
    }
    ensure_parent(cfg.output_json);
    std::ofstream out(cfg.output_json);
    out << std::setprecision(10)
        << "{\n"
        << "  \"backend\": \"adaef\",\n"
        << "  \"phase\": \"online\",\n"
        << "  \"dataset\": " << quote(dataset_stem(cfg.dataset)) << ",\n"
        << "  \"status\": \"ok\",\n"
        << "  \"metric\": " << quote(metric) << ",\n"
        << "  \"metrics\": {\n"
        << "    \"query_count\": " << stats.query_count << ",\n"
        << "    \"achieved_recall\": " << stats.achieved_recall << ",\n"
        << "    \"mean_batch_latency_ms\": " << stats.mean_batch_latency_ms << ",\n"
        << "    \"p50_batch_latency_ms\": " << stats.p50_batch_latency_ms << ",\n"
        << "    \"p95_batch_latency_ms\": " << stats.p95_batch_latency_ms << ",\n"
        << "    \"mean_query_latency_ms\": " << stats.mean_query_latency_ms << ",\n"
        << "    \"mean_latency_ms\": " << stats.mean_query_latency_ms << ",\n"
        << "    \"p50_query_latency_ms\": " << stats.p50_query_latency_ms << ",\n"
        << "    \"p95_query_latency_ms\": " << stats.p95_query_latency_ms << ",\n"
        << "    \"p99_query_latency_ms\": " << stats.p99_query_latency_ms << ",\n"
        << "    \"qps\": " << stats.qps << ",\n"
        << "    \"weighted_average_ef\": " << weighted_average_ef << ",\n"
        << "    \"expected_recall\": " << cfg.expected_recall << ",\n"
        << "    \"statics_length\": " << cfg.statics_length << ",\n"
        << "    \"statics_length_override\": " << (cfg.statics_length_overridden ? 1 : 0) << ",\n"
        << "    \"statics_length_auto\": " << (cfg.statics_length_overridden ? 0 : 1) << ",\n"
        << "    \"warmup_runs\": " << cfg.warmup_runs << ",\n"
        << "    \"measured_runs\": " << cfg.measured_runs << ",\n"
        << "    \"num_threads\": " << cfg.num_threads << ",\n"
        << "    \"parallel_queries\": " << (cfg.parallel_queries ? 1 : 0) << "\n"
        << "  },\n";
    write_artifacts_json(out, paths);
    out << "}\n";
}

void run_build(const Config& cfg, const ArtifactPaths& paths, const std::string& metric) {
    if (!fs::exists(paths.dataset_path)) {
        throw std::runtime_error("Dataset file does not exist: " + paths.dataset_path.string());
    }

    ensure_artifact_dirs(paths);
    auto data = load_train(paths.dataset_path);
    if (metric == "cd") {
        normalize_matrix(*data);
    }

    bool reused_index = false;
    double index_build_ms = 0.0;
    if (fs::exists(paths.index_path) && !cfg.force) {
        reused_index = true;
    } else {
        auto space = hnswdis::init_space(metric, static_cast<int>(data->cols()));
        auto index = std::make_shared<hnswlib::HierarchicalNSW<float>>(
            space.get(),
            static_cast<size_t>(data->rows()),
            static_cast<size_t>(cfg.m),
            static_cast<size_t>(cfg.ef_construction));

        auto start = Clock::now();
        hnswdis::ParallelFor(0, static_cast<size_t>(data->rows()), static_cast<size_t>(cfg.num_threads),
                             [&](size_t row_id, size_t thread_id) {
                                 index->addPoint(static_cast<void*>(data->row(static_cast<Eigen::Index>(row_id)).data()),
                                                 row_id);
                             });
        index_build_ms = elapsed_ms(start, Clock::now());
        index->saveIndex(paths.index_path.string());
    }

    write_build_json(cfg, paths, metric, static_cast<int>(data->rows()), static_cast<int>(data->cols()),
                     index_build_ms, reused_index);
}

void run_offline(const Config& cfg, const ArtifactPaths& paths, const std::string& metric) {
    validate_metric_for_adaef(metric);
    if (!fs::exists(paths.dataset_path)) {
        throw std::runtime_error("Dataset file does not exist: " + paths.dataset_path.string());
    }
    if (!fs::exists(paths.index_path)) {
        throw std::runtime_error("Index file does not exist. Run --phase build first: " + paths.index_path.string());
    }
    ensure_artifact_dirs(paths);

    auto data = load_train(paths.dataset_path);
    if (metric == "cd") {
        normalize_matrix(*data);
    }
    LoadedIndex loaded = load_index(paths.index_path, metric, static_cast<int>(data->cols()));

    double estimator_ms = 0.0;
    double sampling_ms = 0.0;
    double ef_adaptor_ms = 0.0;
    bool estimator_reused = false;
    bool sampling_reused = false;
    bool ef_adaptor_reused = false;

    std::shared_ptr<hnswdis::Estimator> estimator =
        ensure_estimator_file(cfg, paths, *data, metric, estimator_ms, estimator_reused);
    ensure_sampling_file(cfg, paths, *data, metric, sampling_ms, sampling_reused);

    float weighted_average_ef = 0.0f;
    if (fs::exists(paths.ef_adaptor_path) && !cfg.force) {
        hnswdis::EfAdapter adapter(paths.ef_adaptor_path.string());
        weighted_average_ef = adapter.get_wae();
        ef_adaptor_reused = true;
    } else {
        hnswdis::MatrixXf sample_queries;
        hnswdis::MatrixXi sample_ground_truth;
        hnswdis::deserialize_samplings(paths.samplings_path.string(), sample_queries, sample_ground_truth);
        auto sample_queries_ptr = std::make_shared<hnswdis::MatrixXf>(std::move(sample_queries));
        auto sample_ground_truth_ptr = std::make_shared<hnswdis::MatrixXi>(std::move(sample_ground_truth));

        auto start = Clock::now();
        hnswdis::EfAdapter adapter(
            loaded.index,
            data,
            static_cast<size_t>(cfg.k),
            metric,
            cfg.expected_recall,
            cfg.quantile_step,
            cfg.statics_length,
            sample_queries_ptr,
            sample_ground_truth_ptr,
            estimator,
            cfg.ef_upper_bound);
        adapter.serialize(paths.ef_adaptor_path.string());
        ef_adaptor_ms = elapsed_ms(start, Clock::now());
        weighted_average_ef = adapter.get_wae();
        ef_adaptor_reused = false;
    }

    write_offline_json(cfg, paths, metric, estimator_ms, sampling_ms, ef_adaptor_ms,
                       estimator_reused, sampling_reused, ef_adaptor_reused, weighted_average_ef);
}

BatchResult run_query_batch(const Config& cfg,
                            const hnswdis::MatrixXf& queries,
                            const hnswdis::ApproximatedScoreCalculator& score_calculator,
                            const hnswdis::EfAdapter& adapter,
                            const std::shared_ptr<hnswlib::HierarchicalNSW<float>>& index) {
    BatchResult result;
    const int query_count = static_cast<int>(queries.rows());
    result.labels.resize(static_cast<size_t>(query_count));
    result.query_latency_ms.assign(static_cast<size_t>(query_count), 0.0);

    auto batch_start = Clock::now();
    if (cfg.parallel_queries) {
        omp_set_num_threads(cfg.num_threads);
#pragma omp parallel
        {
            hnswdis::Sketch local_sketch(adapter.get_ef_recall_estimators(), adapter.get_expected_recall());
#pragma omp for schedule(static)
            for (int i = 0; i < query_count; ++i) {
                auto query_start = Clock::now();
                auto queue = index->adaptiveSearchKnnTest(
                    queries.row(i).data(),
                    static_cast<size_t>(cfg.k),
                    cfg.statics_length,
                    score_calculator,
                    &local_sketch);
                auto query_end = Clock::now();
                result.labels[static_cast<size_t>(i)] = labels_from_queue(std::move(queue));
                result.query_latency_ms[static_cast<size_t>(i)] = elapsed_ms(query_start, query_end);
            }
        }
    } else {
        hnswdis::Sketch sketch(adapter.get_ef_recall_estimators(), adapter.get_expected_recall());
        for (int i = 0; i < query_count; ++i) {
            auto query_start = Clock::now();
            auto queue = index->adaptiveSearchKnnTest(
                queries.row(i).data(),
                static_cast<size_t>(cfg.k),
                cfg.statics_length,
                score_calculator,
                &sketch);
            auto query_end = Clock::now();
            result.labels[static_cast<size_t>(i)] = labels_from_queue(std::move(queue));
            result.query_latency_ms[static_cast<size_t>(i)] = elapsed_ms(query_start, query_end);
        }
    }
    result.batch_latency_ms = elapsed_ms(batch_start, Clock::now());
    return result;
}

OnlineStats summarize_online(const Config& cfg,
                             const hnswdis::MatrixXi& neighbors,
                             const std::vector<double>& batch_latencies_ms,
                             const std::vector<double>& all_query_latencies_ms,
                             const std::vector<double>& recalls) {
    OnlineStats stats;
    stats.query_count = static_cast<int>(neighbors.rows());
    stats.achieved_recall = mean_value(recalls);
    stats.mean_batch_latency_ms = mean_value(batch_latencies_ms);
    stats.p50_batch_latency_ms = percentile(batch_latencies_ms, 0.50);
    stats.p95_batch_latency_ms = percentile(batch_latencies_ms, 0.95);
    stats.mean_query_latency_ms = mean_value(all_query_latencies_ms);
    stats.p50_query_latency_ms = percentile(all_query_latencies_ms, 0.50);
    stats.p95_query_latency_ms = percentile(all_query_latencies_ms, 0.95);
    stats.p99_query_latency_ms = percentile(all_query_latencies_ms, 0.99);
    const double total_batch_ms = std::accumulate(batch_latencies_ms.begin(), batch_latencies_ms.end(), 0.0);
    if (total_batch_ms > 0.0) {
        stats.qps = (static_cast<double>(neighbors.rows()) * static_cast<double>(cfg.measured_runs)) /
                    (total_batch_ms / 1000.0);
    }
    return stats;
}

void write_per_query_csv(const Config& cfg,
                         const hnswdis::MatrixXf& queries,
                         const hnswdis::MatrixXi& neighbors,
                         const hnswdis::ApproximatedScoreCalculator& score_calculator,
                         hnswdis::EfAdapter& adapter,
                         const std::shared_ptr<hnswlib::HierarchicalNSW<float>>& index) {
    if (cfg.per_query_csv.empty()) {
        return;
    }
    ensure_parent(cfg.per_query_csv);
    std::ofstream out(cfg.per_query_csv);
    out << "dataset,qid,initial_ef,chosen_ef,score,recall,latency_ms\n";

    hnswdis::Sketch sketch(adapter.get_ef_recall_estimators(), adapter.get_expected_recall());
    const size_t initial_ef = std::max(static_cast<size_t>(std::ceil(adapter.get_wae())),
                                       static_cast<size_t>(cfg.k));
    for (int i = 0; i < queries.rows(); ++i) {
        auto query_start = Clock::now();
        auto queue = index->adaptiveSearchKnnTest(
            queries.row(i).data(),
            static_cast<size_t>(cfg.k),
            cfg.statics_length,
            score_calculator,
            &sketch);
        auto query_end = Clock::now();
        std::vector<size_t> labels = labels_from_queue(std::move(queue));

        auto score_queue = index->adaptiveSearchKnn(
            queries.row(i).data(),
            static_cast<size_t>(cfg.k),
            cfg.statics_length,
            score_calculator);
        const float score = score_queue.second;
        size_t chosen_ef = sketch.estimate_ef2(score);
        if (chosen_ef < initial_ef) {
            chosen_ef = initial_ef;
        }

        out << quote(dataset_stem(cfg.dataset)) << ","
            << i << ","
            << initial_ef << ","
            << chosen_ef << ","
            << score << ","
            << recall_one(neighbors, labels, i, cfg.k) << ","
            << elapsed_ms(query_start, query_end) << "\n";
    }
}

void run_online(const Config& cfg, const ArtifactPaths& paths, const std::string& metric) {
    validate_metric_for_adaef(metric);
    if (!fs::exists(paths.dataset_path)) {
        throw std::runtime_error("Dataset file does not exist: " + paths.dataset_path.string());
    }
    if (!fs::exists(paths.index_path)) {
        throw std::runtime_error("Index file does not exist. Run --phase build first: " + paths.index_path.string());
    }
    if (!fs::exists(paths.estimator_path)) {
        throw std::runtime_error("Estimator file does not exist. Run --phase offline first: " + paths.estimator_path.string());
    }
    if (!fs::exists(paths.ef_adaptor_path)) {
        throw std::runtime_error("EF adaptor file does not exist. Run --phase offline first: " + paths.ef_adaptor_path.string());
    }

    auto [queries, data, neighbors] = load_full_dataset(paths.dataset_path);
    if (neighbors->cols() < cfg.k) {
        throw std::runtime_error("neighbors dataset has fewer columns than k");
    }
    if (metric == "cd") {
        normalize_matrix(*data);
        normalize_matrix(*queries);
    }

    LoadedIndex loaded = load_index(paths.index_path, metric, static_cast<int>(data->cols()));
    hnswdis::EfAdapter adapter(paths.ef_adaptor_path.string());
    std::shared_ptr<hnswdis::Estimator> estimator = hnswdis::load_estimator_from_file(paths.estimator_path.string());
    hnswdis::ApproximatedScoreCalculator score_calculator(estimator, cfg.quantile_step);

    const size_t initial_ef = std::max(static_cast<size_t>(std::ceil(adapter.get_wae())),
                                       static_cast<size_t>(cfg.k));
    loaded.index->setEf(initial_ef);

    for (int i = 0; i < cfg.warmup_runs; ++i) {
        (void)run_query_batch(cfg, *queries, score_calculator, adapter, loaded.index);
    }

    std::vector<double> batch_latencies_ms;
    std::vector<double> all_query_latencies_ms;
    std::vector<double> recalls;
    batch_latencies_ms.reserve(static_cast<size_t>(cfg.measured_runs));
    all_query_latencies_ms.reserve(static_cast<size_t>(cfg.measured_runs) * static_cast<size_t>(queries->rows()));
    recalls.reserve(static_cast<size_t>(cfg.measured_runs));

    for (int run = 0; run < cfg.measured_runs; ++run) {
        BatchResult batch = run_query_batch(cfg, *queries, score_calculator, adapter, loaded.index);
        batch_latencies_ms.push_back(batch.batch_latency_ms);
        all_query_latencies_ms.insert(all_query_latencies_ms.end(),
                                      batch.query_latency_ms.begin(),
                                      batch.query_latency_ms.end());
        std::vector<float> per_query_recall =
            hnswdis::compute_recall(*neighbors, batch.labels, static_cast<size_t>(cfg.k), false);
        double run_recall = std::accumulate(per_query_recall.begin(), per_query_recall.end(), 0.0) /
                            static_cast<double>(per_query_recall.size());
        recalls.push_back(run_recall);
    }

    OnlineStats stats = summarize_online(cfg, *neighbors, batch_latencies_ms, all_query_latencies_ms, recalls);
    write_online_json(cfg, paths, metric, stats, adapter.get_wae());
    write_per_query_csv(cfg, *queries, *neighbors, score_calculator, adapter, loaded.index);
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    try {
        cfg = parse_args(argc, argv);
        const std::string metric = infer_metric(cfg);
        if (metric != "cd" && metric != "ipd" && metric != "l2") {
            throw std::invalid_argument("--metric must be auto, cd, ipd, or l2");
        }

        ArtifactPaths paths = build_paths(cfg);
        if (cfg.phase == "build") {
            run_build(cfg, paths, metric);
        } else if (cfg.phase == "offline") {
            run_offline(cfg, paths, metric);
        } else {
            run_online(cfg, paths, metric);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "backend_runner error: " << exc.what() << std::endl;
        write_error_json(cfg.output_json, cfg, exc.what());
        return 2;
    }
}
