#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/DeclarativeRecall.h>

namespace darth1141 {

// Vanilla calls FAISS 1.14.1's native HNSW kernel. ObserveOnly executes the
// ported loop without model calls, while PredictNoET executes the unmodified
// model and interval policy but ignores its termination decision.
enum class Mode { Vanilla, ObserveOnly, PredictNoET, Darth };

struct PredictionCheckpoint {
    int nstep = 0;
    int logical_ndis = 0;
    int insertions = 0;
    int call = 0;
    int interval_before = 0;
    int interval_after = 0;
    float first_nn_distance = -1;
    float predicted_recall = 0;
    bool model_evaluated = false;
    // Corrected inference order matching the trainer's named columns.
    std::array<double, 11> features{};
};

struct QueryDiagnostics {
    // All actual scalar/batch distance evaluations, including the entry point
    // and upper levels. After subtracting upper_distance_evaluations, this may
    // exceed logical_ndis by up to three at a batch-4 early exit.
    std::uint64_t actual_distance_evaluations = 0;
    // Entry point plus upper levels; unavailable (zero) in native Vanilla.
    std::uint64_t upper_distance_evaluations = 0;
    // The legacy predictor's level-0, individually processed neighbor count.
    // Native Vanilla does not expose these counters: -1 means unavailable.
    int logical_ndis = 0;
    int nstep = 0;
    // Diagnostic counters never supplied to the predictor.
    int completed_expansions = 0;
    int started_expansions = 0;
    int unfinished_expansions = 0;
    int insertions = 0;
    int predictor_calls = 0;
    int final_interval = 0;
    float first_nn_distance = -1;
    float predicted_recall = 0;
    bool early_terminated = false;
    bool uses_hashset = false;
    std::vector<PredictionCheckpoint> checkpoints;
    std::vector<faiss::idx_t> expanded_nodes;
};

// Reusable ownership of independent LightGBM worker models only. Query/output
// buffers, logging buffers and execution statistics are NEVER retained here.
// The source model is worker 0, just as in the per-call implementation. While
// this pool is used, the caller must not concurrently search with that source
// model through another pool or through the per-call path, or mutate its model.
// A lease rejects simultaneous searches through this pool instead of allowing
// two native workers to race on LightGBM's per-booster prediction scratch space.
class PredictorPool {
 public:
    class Lease {
     public:
        Lease(Lease&&) = default;
        Lease& operator=(Lease&&) = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        // Copy all current call bindings first, then replace ONLY model
        // ownership. The returned predictor is call-local and dies before the
        // lease, so this pool never keeps pointers to previous search buffers.
        faiss::DARTHPredictorHNSW bind_worker(
                const faiss::DARTHPredictorHNSW& current, int worker) const;

     private:
        friend class PredictorPool;
        Lease(PredictorPool& pool, int workers, std::unique_lock<std::mutex> lock);
        PredictorPool* pool_;
        int workers_;
        std::unique_lock<std::mutex> lock_;
    };

    PredictorPool(const faiss::DARTHPredictorHNSW& source, int capacity);
    PredictorPool(const PredictorPool&) = delete;
    PredictorPool& operator=(const PredictorPool&) = delete;
    int capacity() const;
    Lease acquire(const faiss::DARTHPredictorHNSW& source, int workers);

 private:
    struct WorkerModel {
        BoosterHandle handle;
        std::shared_ptr<void> owner;
    };
    void validate_source(const faiss::DARTHPredictorHNSW& source) const;
    std::vector<WorkerModel> models_;
    std::string source_model_path_;
    std::mutex use_mutex_;
};

struct SearchOptions {
    Mode mode = Mode::Darth;
    int threads = 1;
    std::vector<QueryDiagnostics>* diagnostics = nullptr;
    bool collect_trace = false;
    // nullptr preserves the original per-call model cloning behavior. A pool
    // must outlive this synchronous search and have capacity for min(n,threads).
    PredictorPool* predictor_pool = nullptr;
};

// CPU IndexHNSWFlat, L2/IP, bounded queue only. The index's optional visited
// flag is honored unchanged; its normal nullopt value selects automatically.
// The predictor's data_manager must refer to this call's output arrays.
// Native threads and worker workspaces are created/destroyed inside this call;
// only model handles can persist through options.predictor_pool.
// This isolated build uses the corrected feature contract; model policy is unchanged.
void search(
        faiss::IndexHNSWFlat& index,
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::DARTHPredictorHNSW* predictor,
        const faiss::SearchParametersHNSW* params,
        const SearchOptions& options);

} // namespace darth1141
