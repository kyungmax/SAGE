// Full-run measurement adapter; original DARTH search/feature logic is unchanged.
// Model and input loading precede timing. Every batch allocates its own output,
// copies/normalizes raw queries, binds the predictor and invokes search_DARTH.
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
template<class T> struct Vectors { size_t rows, dim; std::vector<T> values; };
template<class T> Vectors<T> read_vectors(const std::string& name, size_t rows) {
  static_assert(sizeof(T) == 4, "vecs elements are 4 bytes");
  std::ifstream in(name, std::ios::binary);
  int32_t dim = 0;
  in.read(reinterpret_cast<char*>(&dim), 4);
  if (!in || dim <= 0) throw std::runtime_error("Invalid vecs header: " + name);
  in.seekg(0, std::ios::end);
  auto bytes = in.tellg();
  const size_t row_bytes = (size_t(dim) + 1) * 4;
  if (bytes < 0 || size_t(bytes) != rows * row_bytes)
    throw std::runtime_error("Unexpected vecs size: " + name);
  in.seekg(0);
  Vectors<T> result{rows, size_t(dim), std::vector<T>(rows * dim)};
  for (size_t q = 0; q < rows; ++q) {
    int32_t current;
    in.read(reinterpret_cast<char*>(&current), 4);
    in.read(reinterpret_cast<char*>(result.values.data() + q * dim), dim * 4);
    if (!in || current != dim) throw std::runtime_error("Invalid vecs row: " + name);
  }
  return result;
}
template<class T> void write_binary(const std::string& name, const std::vector<T>& v) {
  std::ofstream out(name, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  if (!out) throw std::runtime_error("Cannot write " + name);
}

int main(int argc, char** argv) {
  try {
    const std::set<std::string> allowed{"--index", "--query-file", "--gt-file", "--gt-dist-file",
      "--queries", "--k", "--threads", "--ef", "--initial-interval", "--min-interval",
      "--target", "--model", "--warmups", "--repeats", "--result-prefix"};
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc || !allowed.count(argv[i]) || !args.emplace(argv[i], argv[i + 1]).second)
        throw std::runtime_error("Expected unique supported --name value pairs");
    }
    auto required = [&](const char* key) {
      if (!args.count(key)) throw std::runtime_error(std::string("Missing ") + key);
      return args.at(key);
    };
    auto integer = [&](const char* key, bool zero = false) -> size_t {
      size_t consumed = 0;
      const auto text = required(key);
      const auto n = std::stoll(text, &consumed);
      if (consumed != text.size() || n < (zero ? 0 : 1) || n > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("Invalid integer ") + key);
      return size_t(n);
    };
    const size_t nq = integer("--queries"), k = integer("--k"), threads = integer("--threads"),
      ef = integer("--ef"), initial = integer("--initial-interval"), minimum = integer("--min-interval"),
      warmups = integer("--warmups", true), repeats = integer("--repeats");
    const double target = std::stod(required("--target"));
    if (!(target > 0 && target <= 1) || minimum > initial || ef < k)
      throw std::runtime_error("Invalid target/interval/ef");
    omp_set_dynamic(0);
    omp_set_num_threads(int(threads));
    auto raw = read_vectors<float>(required("--query-file"), nq);
    auto truth = read_vectors<int32_t>(required("--gt-file"), nq);
    auto truth_dist = read_vectors<float>(required("--gt-dist-file"), nq);
    if (truth.dim != k || truth_dist.dim != k) throw std::runtime_error("GT must contain exactly k columns");
    std::vector<faiss::idx_t> gt(truth.values.begin(), truth.values.end());
    std::unique_ptr<faiss::Index> owner(faiss::read_index(required("--index").c_str()));
    auto* index = dynamic_cast<faiss::IndexHNSW*>(owner.get());
    if (!index || index->d != int(raw.dim) || index->ntotal < faiss::idx_t(k) ||
        (index->metric_type != faiss::METRIC_L2 && index->metric_type != faiss::METRIC_INNER_PRODUCT))
      throw std::runtime_error("Expected matching L2/IP HNSW index");
    index->hnsw.efSearch = int(ef);
    const bool normalized = index->metric_type == faiss::METRIC_INNER_PRODUCT;
    auto model = required("--model");
    faiss::DeclarativeRecallDataManager empty;
    faiss::DARTHPredictorHNSW predictor(empty, target, int(initial), int(minimum), false, model.c_str());
    std::vector<faiss::idx_t> reference_labels;
    std::vector<float> reference_distances;
    size_t reference_ndis = 0;
    std::cout << std::setprecision(17);
    for (size_t run = 0; run < warmups + repeats; ++run) {
      faiss::hnsw_stats.reset();
      const auto start = Clock::now();
      std::vector<float> queries(raw.values);
      if (normalized) faiss::fvec_renorm_L2(raw.dim, nq, queries.data());
      std::vector<faiss::idx_t> labels(nq * k, -1);
      std::vector<float> distances(nq * k);
      predictor.data_manager = faiss::DeclarativeRecallDataManager(
        distances.data(), labels.data(), gt.data(), truth_dist.values.data(), nq, raw.dim, k,
        queries.data(), nullptr, nullptr, index->ntotal, gt.data(), k);
      const auto api_start = Clock::now();
      index->search_DARTH(nq, queries.data(), k, distances.data(), labels.data(), predictor);
      const auto end = Clock::now();
      size_t hits = 0;
      for (size_t q = 0; q < nq; ++q) {
        for (size_t j = 0; j < k; ++j) {
          const auto id = labels[q * k + j];
          if (id < 0 || id >= index->ntotal || !std::isfinite(distances[q * k + j]))
            throw std::runtime_error("Invalid search result");
          for (size_t p = 0; p < j; ++p)
            if (labels[q * k + p] == id) throw std::runtime_error("Duplicate result ID");
          hits += std::find(gt.begin() + q * k, gt.begin() + (q + 1) * k, id) != gt.begin() + (q + 1) * k;
        }
      }
      if (run == 0) {
        reference_labels = labels; reference_distances = distances; reference_ndis = faiss::hnsw_stats.ndis;
      } else if (labels != reference_labels || distances.size() != reference_distances.size() ||
                 std::memcmp(distances.data(), reference_distances.data(), distances.size() * sizeof(float)) ||
                 reference_ndis != faiss::hnsw_stats.ndis) {
        throw std::runtime_error("Warmup/repeat results or work differ");
      }
      std::cout << "{\"query_count\":" << nq << ",\"k\":" << k << ",\"threads\":" << threads
        << ",\"ef\":" << ef << ",\"phase\":\"" << (run < warmups ? "warmup" : "timed")
        << "\",\"repeat\":" << (run < warmups ? run + 1 : run - warmups + 1)
        << ",\"target_recall\":" << target << ",\"hits\":" << hits
        << ",\"recall\":" << double(hits) / (nq * k)
        << ",\"ndis\":" << faiss::hnsw_stats.ndis
        << ",\"e2e_seconds\":" << seconds(start, end)
        << ",\"search_api_seconds\":" << seconds(api_start, end) << "}\n" << std::flush;
    }
    const auto prefix = required("--result-prefix");
    write_binary(prefix + ".ids.i64", reference_labels);
    write_binary(prefix + ".distances.f32", reference_distances);
  } catch (const std::exception& e) {
    std::cerr << "DARTH full-run failed: " << e.what() << '\n'; return 1;
  }
}
