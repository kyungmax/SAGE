#pragma once

#include <array>
#include <cstdint>
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
    // The existing predictor's exact inference order, not its training order.
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

struct SearchOptions {
    Mode mode = Mode::Darth;
    int threads = 1;
    std::vector<QueryDiagnostics>* diagnostics = nullptr;
    bool collect_trace = false;
};

// CPU IndexHNSWFlat, L2/IP, bounded queue only. The index's optional visited
// flag is honored unchanged; its normal nullopt value selects automatically.
// The predictor's data_manager must refer to this call's output arrays.
// Worker workspaces/model clones are created and destroyed inside this call.
// Frozen DeclarativeRecall implementation and models are not modified.
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
