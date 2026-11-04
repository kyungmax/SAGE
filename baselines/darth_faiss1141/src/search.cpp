// DARTH's bounded search port, isolated from both its 1.8.0 source and the
// active SAGE build. Search ordering follows the original DARTH kernel;
// native FAISS 1.14.1 supplies distances, upper-level search, heap and visited.
#include <darth1141/search.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>

namespace darth1141 {
namespace {

using Clock = std::chrono::steady_clock;
using faiss::idx_t;
using BlockResultHandler = faiss::HeapBlockResultHandler<faiss::HNSW::C>;

std::unique_ptr<faiss::DistanceComputer> make_distance(const faiss::Index& storage) {
    std::unique_ptr<faiss::DistanceComputer> result(storage.get_distance_computer());
    if (storage.metric_type == faiss::METRIC_INNER_PRODUCT) {
        result = std::make_unique<faiss::NegativeDistanceComputer>(result.release());
    }
    return result;
}

struct CountingDistance final : faiss::DistanceComputer {
    std::unique_ptr<faiss::DistanceComputer> inner;
    std::uint64_t count = 0;

    explicit CountingDistance(const faiss::Index& storage)
            : inner(make_distance(storage)) {}

    void set_query(const float* query) override {
        count = 0;
        inner->set_query(query);
    }

    float operator()(idx_t id) override {
        ++count;
        return (*inner)(id);
    }

    void distances_batch_4(
            idx_t i0, idx_t i1, idx_t i2, idx_t i3,
            float& d0, float& d1, float& d2, float& d3) override {
        count += 4;
        inner->distances_batch_4(i0, i1, i2, i3, d0, d1, d2, d3);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        ++count;
        return inner->symmetric_dis(i, j);
    }
};

std::array<double, 11> snapshot_features(
        faiss::DARTHPredictorHNSW& predictor,
        idx_t query,
        int nstep,
        int ndis,
        int insertions,
        float first_nn) {
    auto& manager = predictor.data_manager;
    // Match the frozen predictor's float-valued accessors and exact order.
    return {double(nstep), double(ndis), double(insertions), double(first_nn),
            double(manager.get_nearest_dist_of_query(query)),
            double(manager.get_avg_dist_of_query(query)),
            double(manager.get_furthest_dist_of_query(query)),
            double(manager.get_variance_of_query(query)),
            double(manager.get_percentile_of_query(query, 0.50)),
            double(manager.get_percentile_of_query(query, 0.25)),
            double(manager.get_percentile_of_query(query, 0.75))};
}

faiss::HNSWStats search_query(
        const faiss::IndexHNSWFlat& index,
        faiss::DistanceComputer& distance,
        const CountingDistance* counted,
        BlockResultHandler::SingleResultHandler& result,
        faiss::VisitedTable& visited,
        idx_t query,
        idx_t k,
        faiss::DARTHPredictorHNSW* predictor,
        const faiss::SearchParametersHNSW* params,
        Mode mode,
        QueryDiagnostics* diagnostic,
        bool collect_trace) {
    const auto& hnsw = index.hnsw;
    faiss::HNSWStats stats;
    if (diagnostic) {
        diagnostic->uses_hashset = visited.visno == 0;
    }
    if (mode == Mode::Vanilla) {
        stats = hnsw.search(distance, &index, result, visited, params);
        if (diagnostic) {
            diagnostic->actual_distance_evaluations = counted->count;
            diagnostic->logical_ndis = -1;
            diagnostic->nstep = -1;
            diagnostic->insertions = -1;
        }
        return stats;
    }
    if (hnsw.entry_point == -1) {
        return stats;
    }

    auto nearest = hnsw.entry_point;
    float nearest_distance = distance(nearest);
    for (int level = hnsw.max_level; level >= 1; --level) {
        stats.combine(faiss::greedy_update_nearest(
                hnsw, distance, level, nearest, nearest_distance));
    }
    if (diagnostic) {
        diagnostic->upper_distance_evaluations = counted->count;
    }

    const int ef_search = params ? params->efSearch : hnsw.efSearch;
    const bool check_relative = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
    const faiss::IDSelector* selector = params ? params->sel : nullptr;
    faiss::MinimaxHeap candidates(std::max(ef_search, int(k)));
    candidates.push(nearest, nearest_distance);

    int ndis = 0;
    int ndis_interval = 0;
    int nstep = 0;
    int total_insertions = 0;
    int prediction_interval = predictor ? predictor->initial_prediction_interval : 0;
    int predictor_calls = 0;
    double predictor_time = 0;
    float first_nn_distance = -1;
    float predicted_recall = 0;
    bool early_terminate = false;
    const bool run_prediction = mode == Mode::PredictNoET || mode == Mode::Darth;
    const double started = predictor ? predictor->data_manager.elapsed_secs() : 0;

    float threshold = result.threshold;
    for (int i = 0; i < candidates.size(); ++i) {
        const idx_t id = candidates.ids[i];
        const float value = candidates.dis[i];
        FAISS_ASSERT(id >= 0);
        if (!selector || selector->is_member(id)) {
            if (value < threshold) {
                ++total_insertions;
                if (result.add_result(value, id)) {
                    threshold = result.threshold;
                }
                // Keep the legacy -1 sentinel semantics, including for IP.
                if (first_nn_distance == -1) {
                    first_nn_distance = value;
                }
            }
        }
        visited.set(id);
    }

    while (candidates.size() > 0) {
        float nearest_value = 0;
        const int expanded = candidates.pop_min(&nearest_value);
        if (check_relative && candidates.count_below(nearest_value) >= ef_search) {
            break;
        }
        if (diagnostic && collect_trace) {
            diagnostic->expanded_nodes.push_back(expanded);
        }

        size_t begin, end;
        hnsw.neighbor_range(expanded, 0, &begin, &end);
        size_t jmax = begin;
        for (size_t j = begin; j < end; ++j) {
            const int id = hnsw.neighbors[j];
            if (id < 0) {
                break;
            }
            visited.prefetch(id);
            ++jmax;
        }
        int buffered = 0;
        size_t saved[4];
        threshold = result.threshold;

        auto add_to_heap = [&](size_t id, float value) {
            bool reached_target = false;
            if (!selector || selector->is_member(id)) {
                if (value < threshold) {
                    ++total_insertions;
                    if (result.add_result(value, id)) {
                        threshold = result.threshold;
                    }
                }
                if (run_prediction) {
                    // Preserve the original integer interval policy, but fail
                    // explicitly instead of executing undefined modulo zero.
                    FAISS_THROW_IF_NOT_MSG(prediction_interval != 0,
                            "DARTH predictor produced a zero interval");
                    if (ndis_interval % prediction_interval == 0) {
                        ++predictor_calls;
                        PredictionCheckpoint checkpoint;
                        if (diagnostic && collect_trace) {
                            checkpoint.nstep = nstep;
                            checkpoint.logical_ndis = ndis;
                            checkpoint.insertions = total_insertions;
                            checkpoint.call = predictor_calls;
                            checkpoint.interval_before = prediction_interval;
                            checkpoint.first_nn_distance = first_nn_distance;
                            checkpoint.model_evaluated =
                                    total_insertions >= predictor->data_manager.k;
                            if (checkpoint.model_evaluated) {
                                checkpoint.features = snapshot_features(*predictor,
                                        query, nstep, ndis, total_insertions,
                                        first_nn_distance);
                            }
                        }
                        predicted_recall = predictor->predict_recall(query,
                                nstep, ndis, total_insertions,
                                predictor->data_manager.elapsed_secs() - started,
                                first_nn_distance, predictor_calls,
                                &prediction_interval, &predictor_time);
                        if (diagnostic && collect_trace) {
                            checkpoint.interval_after = prediction_interval;
                            checkpoint.predicted_recall = predicted_recall;
                            diagnostic->checkpoints.push_back(checkpoint);
                        }
                        reached_target = predicted_recall >= predictor->target_recall;
                        ndis_interval = 0;
                    }
                }
            }
            // The original pushes the last candidate before honoring ET.
            candidates.push(id, value);
            return reached_target && mode == Mode::Darth;
        };

        for (size_t j = begin; j < jmax; ++j) {
            const int id = hnsw.neighbors[j];
            saved[buffered] = id;
            // FAISS 1.14.1 returns whether set() newly visited this id; one
            // hash lookup replaces the old dense-only get()+set() pair.
            buffered += visited.set(id) ? 1 : 0;
            if (buffered == 4) {
                float values[4];
                distance.distances_batch_4(saved[0], saved[1], saved[2], saved[3],
                        values[0], values[1], values[2], values[3]);
                for (size_t part = 0; part < 4; ++part) {
                    ++ndis;
                    ++ndis_interval;
                    early_terminate = add_to_heap(saved[part], values[part]);
                    if (early_terminate) {
                        break;
                    }
                }
                buffered = 0;
            }
            if (early_terminate) {
                break;
            }
        }
        // Deliberately before nstep++: this is the original mid-batch ET.
        if (early_terminate) {
            break;
        }
        for (int part = 0; part < buffered; ++part) {
            const float value = distance(saved[part]);
            ++ndis;
            ++ndis_interval;
            early_terminate = add_to_heap(saved[part], value);
            if (early_terminate) {
                break;
            }
        }
        // Legacy leftover ET increments nstep, unlike a batch-4 ET.
        ++nstep;
        if ((!check_relative && nstep > ef_search) || early_terminate) {
            break;
        }
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;
    stats.nhops += nstep;
    if (predictor) {
        predictor->log_final_recall_result(query, nstep, ndis, total_insertions,
                predictor->data_manager.elapsed_secs() - started,
                first_nn_distance, predicted_recall, prediction_interval,
                predictor_calls, predictor_time);
    }
    if (diagnostic) {
        diagnostic->actual_distance_evaluations = counted->count;
        diagnostic->logical_ndis = ndis;
        diagnostic->nstep = nstep;
        diagnostic->insertions = total_insertions;
        diagnostic->predictor_calls = predictor_calls;
        diagnostic->final_interval = prediction_interval;
        diagnostic->first_nn_distance = first_nn_distance;
        diagnostic->predicted_recall = predicted_recall;
        diagnostic->early_terminated = early_terminate;
    }
    visited.advance();
    return stats;
}

struct WorkerState {
    faiss::VisitedTable visited;
    BlockResultHandler::SingleResultHandler result;
    std::unique_ptr<faiss::DistanceComputer> distance;
    CountingDistance* counted = nullptr;
    faiss::HNSWStats stats;
    idx_t completed = 0;

    WorkerState(const faiss::IndexHNSWFlat& index, BlockResultHandler& results,
            bool diagnostics)
            : visited(index.ntotal, index.hnsw.use_visited_hashset),
              result(results) {
        if (diagnostics) {
            auto counter = std::make_unique<CountingDistance>(*index.storage);
            counted = counter.get();
            distance = std::move(counter);
        } else {
            distance = make_distance(*index.storage);
        }
    }
};

} // namespace

void search(
        faiss::IndexHNSWFlat& index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const faiss::DARTHPredictorHNSW* predictor,
        const faiss::SearchParametersHNSW* params,
        const SearchOptions& options) {
    FAISS_THROW_IF_NOT_MSG(n >= 0 && k > 0 && options.threads > 0,
            "DARTH requires n >= 0, k > 0 and threads > 0");
    FAISS_THROW_IF_NOT_MSG(k <= std::numeric_limits<int>::max(), "k is too large");
    FAISS_THROW_IF_NOT_MSG(n == 0 || (x && distances && labels), "null search buffer");
    FAISS_THROW_IF_NOT_MSG(index.storage && dynamic_cast<faiss::IndexFlat*>(index.storage),
            "DARTH port supports IndexHNSWFlat storage only");
    FAISS_THROW_IF_NOT_MSG(index.metric_type == faiss::METRIC_L2 ||
                    index.metric_type == faiss::METRIC_INNER_PRODUCT,
            "DARTH port supports L2 and inner product only");
    FAISS_THROW_IF_NOT_MSG(index.storage->metric_type == index.metric_type,
            "HNSW and storage metric mismatch");
    FAISS_THROW_IF_NOT_MSG(!index.hnsw.is_panorama, "Panorama is unsupported");
    FAISS_THROW_IF_NOT_MSG(params ? params->bounded_queue : index.hnsw.search_bounded_queue,
            "DARTH port supports the bounded candidate queue only");
    const int ef_search = params ? params->efSearch : index.hnsw.efSearch;
    FAISS_THROW_IF_NOT_MSG(ef_search > 0, "efSearch must be positive");
    switch (options.mode) {
        case Mode::Vanilla:
        case Mode::ObserveOnly:
        case Mode::PredictNoET:
        case Mode::Darth:
            break;
        default:
            FAISS_THROW_MSG("invalid DARTH execution mode");
    }
    const bool requires_predictor = options.mode == Mode::PredictNoET || options.mode == Mode::Darth;
    FAISS_THROW_IF_NOT_MSG(!requires_predictor || predictor, "DARTH mode requires a predictor");
    if (requires_predictor) {
        FAISS_THROW_IF_NOT_MSG(predictor->booster && predictor->initial_prediction_interval > 0,
                "DARTH requires a loaded model and positive initial interval");
    }
    if (predictor && options.mode != Mode::Vanilla) {
        const auto& data = predictor->data_manager;
        FAISS_THROW_IF_NOT_MSG(data.distances == distances && data.labels == labels &&
                        data.nq == n && data.k == k && data.d == index.d,
                "DARTH data_manager must refer to the current search outputs and shape");
    }
    if (options.diagnostics) {
        options.diagnostics->clear();
        options.diagnostics->resize(size_t(n));
    }

    const auto setup_started = Clock::now();
    const int workers_count = int(std::min<idx_t>(n, options.threads));
    std::unique_ptr<faiss::DARTHPredictorHNSW> base;
    if (predictor && options.mode != Mode::Vanilla) {
        base = std::make_unique<faiss::DARTHPredictorHNSW>(*predictor);
        base->prepare_deferred_logs();
    }
    auto* execution = base ? base->execution_stats : nullptr;
    if (execution) {
        *execution = faiss::HNSWSearchExecutionStats{};
        execution->requested_threads = options.threads;
        execution->worker_count = workers_count;
    }
    std::vector<faiss::DARTHPredictorHNSW> predictors;
    if (base) {
        predictors.reserve(workers_count);
        for (int worker = 0; worker < workers_count; ++worker) {
            predictors.push_back(worker == 0 ? *base : base->clone_for_worker());
        }
    }
    BlockResultHandler results(size_t(n), distances, labels, size_t(k));
    std::vector<std::unique_ptr<WorkerState>> states;
    states.reserve(workers_count);
    for (int worker = 0; worker < workers_count; ++worker) {
        states.push_back(std::make_unique<WorkerState>(index, results,
                options.diagnostics != nullptr));
    }
    const auto search_started = Clock::now();
    if (execution) {
        execution->setup_seconds = std::chrono::duration<double>(search_started - setup_started).count();
    }

    std::atomic<idx_t> next{workers_count};
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const idx_t interrupt_period = std::max<idx_t>(1,
            faiss::InterruptCallback::get_period_hint(index.hnsw.max_level * index.d * ef_search));

    auto worker_body = [&](int worker) {
        const int old_threads = omp_get_max_threads();
        omp_set_num_threads(1);
        try {
            auto& state = *states[worker];
            idx_t since_interrupt = 0;
            auto answer = [&](idx_t query) {
                if (failed.load(std::memory_order_relaxed)) {
                    return;
                }
                if (since_interrupt == 0 && faiss::InterruptCallback::is_interrupted()) {
                    FAISS_THROW_MSG("computation interrupted");
                }
                state.result.begin(query);
                state.distance->set_query(x + query * index.d);
                state.stats.combine(search_query(index, *state.distance, state.counted,
                        state.result, state.visited, query, k,
                        base ? &predictors[worker] : nullptr, params, options.mode,
                        options.diagnostics ? &(*options.diagnostics)[size_t(query)] : nullptr,
                        options.collect_trace));
                state.result.end();
                ++state.completed;
                if (++since_interrupt >= interrupt_period) {
                    since_interrupt = 0;
                }
            };
            answer(worker);
            while (!failed.load(std::memory_order_relaxed)) {
                const idx_t start = next.fetch_add(8, std::memory_order_relaxed);
                if (start >= n) {
                    break;
                }
                for (idx_t query = start; query < std::min(start + 8, n); ++query) {
                    answer(query);
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(failure_mutex);
            if (!failure) {
                failure = std::current_exception();
            }
            failed.store(true, std::memory_order_relaxed);
        }
        omp_set_num_threads(old_threads);
    };

    std::vector<std::thread> workers;
    workers.reserve(workers_count);
    try {
        // Keep the original native-thread design: LightGBM single-row scratch
        // space must not receive an outer OpenMP worker id.
        if (workers_count == 1 && !omp_in_parallel()) {
            worker_body(0);
        } else {
            for (int worker = 0; worker < workers_count; ++worker) {
                workers.emplace_back(worker_body, worker);
            }
        }
    } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        for (auto& worker : workers) {
            worker.join();
        }
        throw;
    }
    for (auto& worker : workers) {
        worker.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    faiss::InterruptCallback::check();
    faiss::HNSWStats total;
    for (const auto& state : states) {
        total.combine(state->stats);
        if (execution && state->completed) {
            ++execution->active_workers;
        }
    }
    faiss::hnsw_stats.combine(total);
    const auto logging_started = Clock::now();
    if (execution) {
        execution->search_seconds = std::chrono::duration<double>(logging_started - search_started).count();
    }
    if (base) {
        base->flush_deferred_logs();
    }
    if (execution) {
        execution->logging_seconds = std::chrono::duration<double>(Clock::now() - logging_started).count();
    }
    if (index.metric_type == faiss::METRIC_INNER_PRODUCT) {
        for (size_t i = 0; i < size_t(n) * size_t(k); ++i) {
            distances[i] = -distances[i];
        }
    }
}

} // namespace darth1141
