// Paper Algorithm 2 (arXiv:2505.15636v1, p.17), on a read-only FAISS HNSW graph.
// Reference: the local arxiv_paper_comparison/native_abs.cpp prototype.
// This adapter has no ef parameter, global gamma, or dependency on author code.
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/VisitedTable.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
using Node = std::pair<float, faiss::HNSW::storage_idx_t>;
enum Stat { NDIS, UPPER, LEVEL0, EXPANDED, POPPED, PEAK, BOUND, ENTRY, NCOLS };

struct CountingDistance : faiss::DistanceComputer {
    std::unique_ptr<faiss::DistanceComputer> inner;
    uint64_t count = 0;
    explicit CountingDistance(const faiss::IndexHNSW& index) {
        inner.reset(index.storage->get_distance_computer());
        if (index.metric_type == faiss::METRIC_INNER_PRODUCT)
            inner.reset(new faiss::NegativeDistanceComputer(inner.release()));
    }
    void set_query(const float* query) override {
        count = 0;
        inner->set_query(query);
    }
    float operator()(faiss::idx_t id) override { ++count; return (*inner)(id); }
    float symmetric_dis(faiss::idx_t a, faiss::idx_t b) override {
        return inner->symmetric_dis(a, b);
    }
    void distances_batch_4(faiss::idx_t a, faiss::idx_t b, faiss::idx_t c,
                           faiss::idx_t d, float& da, float& db, float& dc, float& dd) override {
        count += 4;
        inner->distances_batch_4(a, b, c, d, da, db, dc, dd);
    }
};

struct Workspace {
    CountingDistance distance;
    faiss::VisitedTable visited;
    std::vector<Node> candidates, best;
    explicit Workspace(const faiss::IndexHNSW& index)
        : distance(index), visited(index.ntotal, index.hnsw.use_visited_hashset) {}
};

struct Context {
    const faiss::IndexHNSWFlat* index;
    faiss::idx_t ntotal;
    int dimension, threads;
    std::mutex mutex;
    std::vector<std::unique_ptr<Workspace>> work;
    Context(const faiss::Index* candidate, int nthreads, bool normalized_ip)
        : index(dynamic_cast<const faiss::IndexHNSWFlat*>(candidate)), threads(nthreads) {
        if (!index || !index->storage || index->ntotal < 1 || threads < 1)
            throw std::invalid_argument("A nonempty IndexHNSWFlat and positive threads are required");
        if (index->metric_type != faiss::METRIC_L2 &&
            !(index->metric_type == faiss::METRIC_INNER_PRODUCT && normalized_ip))
            throw std::invalid_argument("Use L2 or explicitly validated unit-normalized IP");
        ntotal = index->ntotal;
        dimension = index->d;
        if (index->hnsw.levels.size() != size_t(ntotal) ||
            index->hnsw.offsets.size() != size_t(ntotal) + 1)
            throw std::invalid_argument("HNSW graph size does not match the vector storage");
        for (int t = 0; t < threads; ++t) work.push_back(std::make_unique<Workspace>(*index));
    }
};

// For normalized IP, raw is -IP and 1+raw is half squared chord distance.
// The factor 2 cancels in the ratio. Returned distances retain FAISS semantics.
double radial(float raw, bool ip) {
    double value = ip ? 1.0 + double(raw) : double(raw);
    if (!std::isfinite(value) || value < (ip ? -5e-5 : 0.0))
        throw std::runtime_error("Non-finite/invalid distance; check vectors and normalization");
    return std::max(0.0, value);
}

void search_level0(const faiss::IndexHNSWFlat& index, Workspace& w,
                   Node start, int k, double multiplier,
                   float* distances, faiss::idx_t* labels, uint64_t* stats) {
    bool ip = index.metric_type == faiss::METRIC_INNER_PRODUCT;
    auto& candidates = w.candidates;
    auto& best = w.best;
    candidates.clear(); best.clear();
    radial(start.first, ip);
    candidates.push_back(start);
    best.push_back(start);
    w.visited.set(start.second);
    stats[PEAK] = 1;
    auto insert = [&](faiss::HNSW::storage_idx_t id, float raw) {
        double distance = radial(raw, ip);
        // Algorithm 2, line 11: STRICT admission when the result heap is full.
        if (best.size() < size_t(k) ||
            distance < multiplier * radial(best.front().first, ip)) {
            candidates.emplace_back(raw, id);
            std::push_heap(candidates.begin(), candidates.end(), std::greater<Node>());
            best.emplace_back(raw, id);
            std::push_heap(best.begin(), best.end());
            if (best.size() > size_t(k)) {
                std::pop_heap(best.begin(), best.end());
                best.pop_back();
            }
            stats[PEAK] = std::max(stats[PEAK], uint64_t(candidates.size()));
        }
    };
    while (!candidates.empty()) {
        std::pop_heap(candidates.begin(), candidates.end(), std::greater<Node>());
        Node current = candidates.back();
        candidates.pop_back();
        ++stats[POPPED];
        // Algorithm 2, line 6: equality terminates, including gamma=0.
        if (best.size() == size_t(k) &&
            radial(current.first, ip) >= multiplier * radial(best.front().first, ip)) {
            stats[BOUND] = 1;
            break;
        }
        ++stats[EXPANDED];
        size_t begin, end;
        index.hnsw.neighbor_range(current.second, 0, &begin, &end);
        faiss::HNSW::storage_idx_t buffered[4];
        int count = 0;
        for (size_t j = begin; j < end; ++j) {
            auto id = index.hnsw.neighbors[j];
            if (id < 0) break;
            if (id >= index.ntotal) throw std::runtime_error("Out-of-range neighbor ID");
            if (!w.visited.set(id)) continue;
            buffered[count++] = id;
            if (count == 4) {
                float ds[4];
                w.distance.distances_batch_4(buffered[0], buffered[1], buffered[2], buffered[3],
                                             ds[0], ds[1], ds[2], ds[3]);
                // Preserve adjacency order: update B after EACH neighbor.
                for (int t = 0; t < 4; ++t) insert(buffered[t], ds[t]);
                count = 0;
            }
        }
        for (int t = 0; t < count; ++t) insert(buffered[t], w.distance(buffered[t]));
    }
    std::sort(best.begin(), best.end());
    for (int j = 0; j < k; ++j) {
        bool present = j < int(best.size());
        labels[j] = present ? best[j].second : -1;
        distances[j] = present ? (ip ? -best[j].first : best[j].first)
                              : (ip ? -INFINITY : INFINITY);
    }
}
} // namespace

extern "C" int paper_abs_api_version() { return 1; }
extern "C" const char* paper_abs_last_error() { return last_error.c_str(); }
extern "C" void* paper_abs_create(const faiss::Index* index, int threads, int normalized_ip) {
    try { return new Context(index, threads, normalized_ip != 0); }
    catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
extern "C" void paper_abs_destroy(void* context) { delete static_cast<Context*>(context); }

extern "C" int paper_abs_search(void* opaque, faiss::idx_t n, const float* queries,
                               int k, double gamma, faiss::idx_t forced_entry,
                               float* distances, faiss::idx_t* labels, uint64_t* statistics) {
    try {
        auto* context = static_cast<Context*>(opaque);
        if (!context) throw std::invalid_argument("ABS context is closed");
        std::lock_guard<std::mutex> lock(context->mutex);
        const auto& index = *context->index;
        if (index.ntotal != context->ntotal || index.d != context->dimension)
            throw std::runtime_error("Index changed; recreate the ABS context");
        if (n < 0 || k < 1 || k > index.ntotal || !std::isfinite(gamma) || gamma < 0 ||
            !std::isfinite((1.0 + gamma) * (1.0 + gamma)) ||
            forced_entry < -1 || forced_entry >= index.ntotal ||
            (n && (!queries || !distances || !labels || !statistics)))
            throw std::invalid_argument("Invalid ABS search arguments");
        if (!n) return 0;
        const double multiplier = (1.0 + gamma) * (1.0 + gamma);
        std::exception_ptr error;
        std::atomic<bool> failed{false};
#pragma omp parallel num_threads(context->threads)
        {
            auto& w = *context->work[omp_get_thread_num()];
#pragma omp for schedule(dynamic)
            for (faiss::idx_t i = 0; i < n; ++i) {
                if (failed.load()) continue;
                try {
                    auto* stats = statistics + i * NCOLS;
                    std::fill(stats, stats + NCOLS, uint64_t(0));
                    w.distance.set_query(queries + i * index.d);
                    auto entry = forced_entry >= 0
                        ? faiss::HNSW::storage_idx_t(forced_entry) : index.hnsw.entry_point;
                    if (entry < 0 || entry >= index.ntotal)
                        throw std::runtime_error("Missing or invalid HNSW entry point");
                    float entry_distance = w.distance(entry);
                    if (forced_entry < 0 && index.hnsw.max_level > 0) {
                        for (int level = index.hnsw.max_level; level >= 1; --level)
                            faiss::greedy_update_nearest(index.hnsw, w.distance, level,
                                                         entry, entry_distance);
                        stats[UPPER] = w.distance.count;
                    }
                    stats[ENTRY] = entry;
                    // Reuse the entry distance from upper descent; do not compute it twice.
                    search_level0(index, w, {entry_distance, entry}, k, multiplier,
                                  distances + i * k, labels + i * k, stats);
                    stats[NDIS] = w.distance.count;
                    stats[LEVEL0] = stats[NDIS] - stats[UPPER];
                    w.visited.advance();
                } catch (...) {
                    w.visited.advance();
#pragma omp critical(paper_abs_errors)
                    { if (!error) error = std::current_exception(); }
                    failed = true;
                }
            }
        }
        if (error) std::rethrow_exception(error);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
