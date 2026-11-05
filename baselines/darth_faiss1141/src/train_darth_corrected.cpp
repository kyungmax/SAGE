// Generate observations with the original 1.8 DARTH search and isolated corrected
// furthest-distance helper. CSV order, formatting and logging intervals are unchanged.
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

double elapsed(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

template <class T> struct Vectors {
  size_t rows;
  size_t dim;
  std::vector<T> values;
};

template <class T>
Vectors<T> read_vectors(const std::string& path, size_t rows) {
  static_assert(sizeof(T) == 4, "vecs elements are 4 bytes");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open " + path);
  int32_t dimension;
  input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
  if (!input || dimension <= 0)
    throw std::runtime_error("Invalid vecs dimension: " + path);
  input.seekg(0, std::ios::end);
  const auto bytes = input.tellg();
  const size_t row_bytes = (size_t(dimension) + 1) * sizeof(T);
  if (bytes < 0 || size_t(bytes) % row_bytes || size_t(bytes) / row_bytes < rows)
    throw std::runtime_error("Invalid vecs file size: " + path);
  if (rows > std::numeric_limits<size_t>::max() / size_t(dimension))
    throw std::runtime_error("Vector allocation size overflow");
  input.seekg(0);
  Vectors<T> result{rows, size_t(dimension),
                    std::vector<T>(rows * size_t(dimension))};
  for (size_t row = 0; row < rows; ++row) {
    int32_t current_dimension;
    input.read(reinterpret_cast<char*>(&current_dimension), sizeof(current_dimension));
    if (!input || current_dimension != dimension)
      throw std::runtime_error("Inconsistent vecs dimension: " + path);
    input.read(reinterpret_cast<char*>(result.values.data() + row * dimension),
               size_t(dimension) * sizeof(T));
    if (!input) throw std::runtime_error("Truncated vecs file: " + path);
  }
  return result;
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "Collect the original DARTH training observations with corrected distance features.\n"
                << "Required: --index PATH --query-file FVECS --gt-file IVECS\n"
                << "          --gt-dist-file FVECS --queries N --k K --ef E\n"
                << "          --threads T --logging-interval N --output CSV_OR_PIPE\n"
                << "Inner-product queries are normalized; L2 queries are used unchanged.\n";
      return 0;
    }
    const auto process_start = Clock::now();
    const std::set<std::string> allowed{
        "--index", "--query-file", "--gt-file", "--gt-dist-file", "--queries",
        "--k", "--ef", "--threads", "--logging-interval", "--output"};
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc || !allowed.count(argv[i]))
        throw std::runtime_error("Expected supported --name value pairs");
      if (!args.emplace(argv[i], argv[i + 1]).second)
        throw std::runtime_error("Duplicate argument: " + std::string(argv[i]));
    }
    const auto required = [&](const char* key) -> std::string {
      const auto found = args.find(key);
      if (found == args.end())
        throw std::runtime_error(std::string("Missing ") + key);
      return found->second;
    };
    const auto positive = [&](const char* key) -> size_t {
      const auto value = required(key);
      size_t end;
      const auto number = std::stoll(value, &end);
      if (number <= 0 || end != value.size())
        throw std::runtime_error(std::string("Invalid positive integer: ") + key);
      return size_t(number);
    };
    const auto nq = positive("--queries");
    const auto k = positive("--k");
    const auto ef = positive("--ef");
    const auto threads = positive("--threads");
    const auto logging_interval = positive("--logging-interval");
    if (ef < k || k > size_t(std::numeric_limits<int>::max()) ||
        nq > size_t(std::numeric_limits<faiss::idx_t>::max()) ||
        threads > size_t(std::numeric_limits<int>::max()) ||
        ef > size_t(std::numeric_limits<int>::max()) ||
        logging_interval > size_t(std::numeric_limits<int>::max()) ||
        nq > std::numeric_limits<size_t>::max() / k)
      throw std::runtime_error("Search parameter out of range");
    omp_set_dynamic(0);
    omp_set_num_threads(int(threads));

    const auto input_start = Clock::now();
    auto queries = read_vectors<float>(required("--query-file"), nq);
    auto gt_raw = read_vectors<int32_t>(required("--gt-file"), nq);
    auto gt_dist_raw = read_vectors<float>(required("--gt-dist-file"), nq);
    if (gt_raw.dim < k || gt_dist_raw.dim < k)
      throw std::runtime_error("Ground truth has fewer than k neighbors");
    const auto input_end = Clock::now();
    std::unique_ptr<faiss::Index> owner(faiss::read_index(required("--index").c_str()));
    const auto index_end = Clock::now();
    auto* index = dynamic_cast<faiss::IndexHNSWFlat*>(owner.get());
    if (!index || index->d != int(queries.dim) || index->ntotal < faiss::idx_t(k))
      throw std::runtime_error("Expected compatible HNSWFlat index");
    if (!index->hnsw.search_bounded_queue || index->hnsw.upper_beam != 1)
      throw std::runtime_error("Training requires bounded HNSW search with upper_beam=1");
    if (index->metric_type != faiss::METRIC_INNER_PRODUCT &&
        index->metric_type != faiss::METRIC_L2)
      throw std::runtime_error("Expected L2 or inner-product index");
    if (std::any_of(queries.values.begin(), queries.values.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(gt_dist_raw.values.begin(), gt_dist_raw.values.end(),
                    [](float value) { return !std::isfinite(value); }))
      throw std::runtime_error("Non-finite query or ground-truth distance");
    if (std::any_of(gt_raw.values.begin(), gt_raw.values.end(),
                    [&](int32_t id) { return id < 0 || id >= index->ntotal; }))
      throw std::runtime_error("Ground-truth ID outside the indexed corpus");
    index->hnsw.efSearch = int(ef);
    const bool normalized = index->metric_type == faiss::METRIC_INNER_PRODUCT;
    const auto normalization_start = Clock::now();
    if (normalized)
      faiss::fvec_renorm_L2(queries.dim, nq, queries.values.data());
    const auto normalization_end = Clock::now();
    std::vector<faiss::idx_t> gt_all(gt_raw.values.begin(), gt_raw.values.end());
    std::vector<faiss::idx_t> gt(nq * k), labels(nq * k, -1);
    std::vector<float> gt_dist(nq * k), distances(nq * k, 0);
    for (size_t q = 0; q < nq; ++q) {
      std::copy_n(gt_all.data() + q * gt_raw.dim, k, gt.data() + q * k);
      std::copy_n(gt_dist_raw.values.data() + q * gt_dist_raw.dim, k,
                  gt_dist.data() + q * k);
    }
    auto output = required("--output");
    faiss::DeclarativeRecallDataManager manager(
        distances.data(), labels.data(), gt.data(), gt_dist.data(), nq,
        queries.dim, k, queries.values.data(), output.data(), nullptr,
        index->ntotal, gt_all.data(), gt_raw.dim);
    faiss::DeclarativeRecallDataCollectorHNSW collector(manager, int(logging_interval), 0);
    // The original HNSW logger already emits only aggregate features; unlike
    // the IVF logger, it never emits raw query dimensions. No feature is dropped.
    collector.include_data_dimensions = false;
    collector.init_log_file();
    struct stat output_stat;
    if (fstat(fileno(collector.log_file), &output_stat) != 0)
      throw std::runtime_error("Cannot inspect training trace output");
    const bool output_is_fifo = S_ISFIFO(output_stat.st_mode);
    faiss::hnsw_stats.reset();
    const auto trace_start = Clock::now();
    index->search_declarative_recall_data_generation(
        nq, queries.values.data(), k, distances.data(), labels.data(), collector);
    const auto trace_end = Clock::now();
    if (std::fflush(collector.log_file) != 0 || std::ferror(collector.log_file))
      throw std::runtime_error("Failed writing training trace: " + output);
    const int close_result = std::fclose(collector.log_file);
    collector.log_file = nullptr;
    if (close_result != 0)
      throw std::runtime_error("Failed closing training trace: " + output);

    size_t hits = 0;
    for (size_t q = 0; q < nq; ++q) {
      for (size_t j = 0; j < k; ++j) {
        const auto label = labels[q * k + j];
        if (label < 0 || label >= index->ntotal || !std::isfinite(distances[q * k + j]))
          throw std::runtime_error("Trace search produced an invalid result");
        for (size_t z = 0; z < j; ++z)
          if (label == labels[q * k + z])
            throw std::runtime_error("Trace search produced duplicate result IDs");
        hits += std::find(gt.begin() + q * k, gt.begin() + (q + 1) * k, label) !=
            gt.begin() + (q + 1) * k;
      }
    }
    const auto count_start = Clock::now();
    size_t lines = 0;
    if (!output_is_fifo) {
      std::ifstream trace(output, std::ios::binary);
      if (!trace) throw std::runtime_error("Cannot reopen training trace");
      std::vector<char> buffer(1024 * 1024);
      while (trace.read(buffer.data(), buffer.size()) || trace.gcount())
        lines += std::count(buffer.begin(), buffer.begin() + trace.gcount(), '\n');
      if (!trace.eof() || lines <= 1)
        throw std::runtime_error("Invalid or empty training trace");
    }
    const auto count_end = Clock::now();
    std::cout << std::setprecision(17)
              << "\n{\"query_count\":" << nq << ",\"k\":" << k
              << ",\"threads\":" << threads << ",\"ef\":" << ef
              << ",\"logging_interval\":" << logging_interval
              << ",\"trace_rows\":" << (output_is_fifo ? std::string("null") : std::to_string(lines - 1))
              << ",\"output_is_fifo\":" << (output_is_fifo ? "true" : "false")
              << ",\"ndis\":" << faiss::hnsw_stats.ndis
              << ",\"mean_ndis\":" << double(faiss::hnsw_stats.ndis) / nq
              << ",\"recall_hits\":" << hits
              << ",\"recall\":" << double(hits) / double(nq * k)
              << ",\"normalized_queries\":" << (normalized ? "true" : "false")
              << ",\"include_data_dimensions\":false"
              << ",\"input_load_seconds\":" << elapsed(input_start, input_end)
              << ",\"index_load_seconds\":" << elapsed(input_end, index_end)
              << ",\"normalization_seconds\":" << elapsed(normalization_start, normalization_end)
              << ",\"trace_api_seconds\":" << elapsed(trace_start, trace_end)
              << ",\"trace_count_seconds\":" << elapsed(count_start, count_end)
              << ",\"total_seconds\":" << elapsed(process_start, count_end)
              << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "Original DARTH trace generation failed: " << error.what() << '\n';
    return 1;
  }
}
