/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>

#include <cstdint>
#include "faiss/Index.h"

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

namespace faiss {

using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;

HNSWStats hnsw_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

static inline float sage_external_distance(
        float raw_distance,
        bool similarity_metric) {
    return similarity_metric ? (1.0f + raw_distance) : raw_distance;
}

static inline float sage_rank_distance_or_nan(
        const std::priority_queue<HNSW::Node>& top_candidates,
        size_t rank,
        bool similarity_metric) {
    if (rank == 0 || top_candidates.size() < rank) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    auto snapshot = top_candidates;
    std::vector<HNSW::Node> sorted;
    sorted.reserve(snapshot.size());
    while (!snapshot.empty()) {
        sorted.push_back(snapshot.top());
        snapshot.pop();
    }
    std::sort(sorted.begin(), sorted.end(), [](const HNSW::Node& a, const HNSW::Node& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    return sage_external_distance(sorted[rank - 1].first, similarity_metric);
}

static inline HNSW::storage_idx_t sage_resolve_entry_point(
        const IndexHNSW& index,
        idx_t hidden_label) {
    const HNSW& hnsw = index.hnsw;
    if (index.ntotal <= 0 || hnsw.entry_point < 0) {
        return -1;
    }
    if (hidden_label < 0 || hnsw.entry_point != hidden_label) {
        return hnsw.entry_point;
    }

    HNSW::storage_idx_t fallback = -1;
    int fallback_level = -1;
    for (idx_t i = 0; i < index.ntotal; i++) {
        if (i == hidden_label) {
            continue;
        }
        const int level = hnsw.levels[i];
        if (level > fallback_level) {
            fallback = static_cast<HNSW::storage_idx_t>(i);
            fallback_level = level;
        }
    }
    return fallback;
}

static inline size_t sage_greedy_update_nearest_hidden(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        HNSW::storage_idx_t& nearest,
        float& d_nearest,
        idx_t hidden_label) {
    size_t ndis = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        size_t begin = 0, end = 0;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            HNSW::storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) {
                break;
            }
            if (v == hidden_label) {
                continue;
            }
            float dis = qdis(v);
            ndis++;
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
                changed = true;
            }
        }
    }
    return ndis;
}

struct SageEntryPointResult {
    HNSW::storage_idx_t nearest = -1;
    float distance = std::numeric_limits<float>::infinity();
    size_t distance_count = 0;
};

static SageEntryPointResult sage_resolve_search_entry_hidden(
        const IndexHNSW& index,
        DistanceComputer& qdis,
        idx_t hidden_label) {
    SageEntryPointResult result;
    const HNSW& hnsw = index.hnsw;
    result.nearest = sage_resolve_entry_point(index, hidden_label);
    if (result.nearest < 0 || result.nearest == hidden_label) {
        return result;
    }

    result.distance = qdis(result.nearest);
    result.distance_count = 1;
    int start_level = hnsw.max_level;
    if (result.nearest != hnsw.entry_point) {
        start_level = std::min(
                start_level,
                std::max(0, hnsw.levels[result.nearest] - 1));
    }
    for (int level = start_level; level >= 1; level--) {
        result.distance_count += sage_greedy_update_nearest_hidden(
                hnsw,
                qdis,
                level,
                result.nearest,
                result.distance,
                hidden_label);
    }
    return result;
}

struct SageTraceOutputs {
    idx_t max_steps = 0;
    uint64_t* step_counts = nullptr;
    uint64_t* truncated_flags = nullptr;
    uint64_t* distance_counts = nullptr;
    float* closest_dists = nullptr;
    idx_t* node_labels = nullptr;
    uint64_t* rs_sizes = nullptr;
    uint64_t* rs_sizes_after = nullptr;
    uint64_t* is_full_pop_after = nullptr;
    uint64_t* full_pop_counts_after = nullptr;
    uint64_t* popped_degrees = nullptr;
    uint64_t* unvisited_counts = nullptr;
    uint64_t* accepted_counts = nullptr;
    float* runtime_accepted_rates = nullptr;
    float* runtime_cfrs = nullptr;
    float* runtime_smoothed_cfrs = nullptr;
    float* internal_dists = nullptr;
    float* popped_query_dists = nullptr;
    float* furthest_dists = nullptr;
    float* best_dists = nullptr;
    float* top_k_dists = nullptr;
    float* ef_half_dists = nullptr;
    float* ef_quarter_dists = nullptr;
    float* sqrt_ef_dists = nullptr;
    float* top_2k_dists = nullptr;
    float* top_3k_dists = nullptr;
};

struct SageLevel0SearchResult {
    size_t distance_count = 0;
    size_t hop_count = 0;
    bool exhausted = false;
    bool truncated = false;
    size_t stored_steps = 0;
    size_t full_pop_count = 0;
    size_t window_obs_count = 0;
    float mean_smoothed_cfr_classify_window =
            std::numeric_limits<float>::quiet_NaN();
    bool usable_for_mean_window = false;
    float closest_dist = std::numeric_limits<float>::infinity();
};

struct SageCfrWindowConfig {
    int classify_start = 4;
    int classify_end = 16;
    float cfr_ema_decay = 0.8f;
};

static SageCfrWindowConfig sage_resolve_cfr_window_config(
        const SearchParameters* params) {
    SageCfrWindowConfig config;
    if (params) {
        const auto adaptive_params =
                dynamic_cast<const SearchParametersHNSWAdaptiveLight*>(params);
        if (adaptive_params) {
            config.classify_start = adaptive_params->classify_start;
            config.classify_end = adaptive_params->classify_end;
            config.cfr_ema_decay = adaptive_params->cfr_ema_decay;
        }
    }
    FAISS_THROW_IF_NOT_FMT(
            config.classify_start >= 0,
            "classify_start must be >= 0, got %d",
            config.classify_start);
    FAISS_THROW_IF_NOT_FMT(
            config.classify_end >= config.classify_start &&
                    config.classify_end >= 1,
            "classify_end must be >= max(classify_start, 1), got start=%d end=%d",
            config.classify_start,
            config.classify_end);
    FAISS_THROW_IF_NOT_MSG(
            std::isfinite(config.cfr_ema_decay) &&
                    config.cfr_ema_decay >= 0.0f &&
                    config.cfr_ema_decay <= 1.0f,
            "cfr_ema_decay must be finite and lie in [0, 1]");
    return config;
}

static SageLevel0SearchResult sage_search_level0_hidden(
        const IndexHNSW& index,
        DistanceComputer& qdis,
        VisitedTable& vt,
        HNSW::storage_idx_t nearest,
        float d_nearest,
        idx_t k,
        idx_t ef,
        idx_t hidden_label,
        bool similarity_metric,
        const SearchParameters* params,
        float* out_distances,
        idx_t* out_labels,
        SageTraceOutputs* trace,
        idx_t trace_row) {
    SageLevel0SearchResult result;
    const HNSW& hnsw = index.hnsw;
    const IDSelector* sel = params ? params->sel : nullptr;
    const SageCfrWindowConfig cfr_config =
            sage_resolve_cfr_window_config(params);
    const float cfr_ema_update = 1.0f - cfr_config.cfr_ema_decay;
    const size_t normalized_k = std::max<size_t>(static_cast<size_t>(k), 1);
    const size_t ef_search = std::max<size_t>(static_cast<size_t>(ef), normalized_k);
    const float nan = std::numeric_limits<float>::quiet_NaN();

    std::priority_queue<HNSW::Node> top_candidates;
    std::priority_queue<HNSW::Node, std::vector<HNSW::Node>, std::greater<HNSW::Node>>
            candidate_set;

    if (hidden_label >= 0 && hidden_label < index.ntotal) {
        vt.set(static_cast<HNSW::storage_idx_t>(hidden_label));
    }
    vt.set(nearest);
    candidate_set.emplace(d_nearest, nearest);
    if (!sel || sel->is_member(nearest)) {
        top_candidates.emplace(d_nearest, nearest);
    }

    float best_raw_distance = d_nearest;
    float smoothed_cfr_ema = nan;
    size_t full_pop_count = 0;
    size_t stored_steps = 0;
    double classify_window_sum = 0.0;
    size_t classify_window_count = 0;
    bool truncated = false;

    while (!candidate_set.empty()) {
        const float candidate_distance = candidate_set.top().first;
        const HNSW::storage_idx_t current = candidate_set.top().second;
        const float current_lower_bound = top_candidates.empty()
                ? std::numeric_limits<float>::max()
                : top_candidates.top().first;
        if (top_candidates.size() == ef_search &&
            candidate_distance > current_lower_bound) {
            break;
        }

        candidate_set.pop();
        result.hop_count++;

        const size_t rs_size_before = top_candidates.size();
        const float raw_internal_before = current_lower_bound;
        size_t begin = 0, end = 0;
        hnsw.neighbor_range(current, 0, &begin, &end);

        size_t popped_degree = 0;
        size_t unvisited_count = 0;
        size_t accepted_count = 0;
        float lower_bound = current_lower_bound;

        for (size_t j = begin; j < end; j++) {
            const HNSW::storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) {
                break;
            }
            if (v == hidden_label) {
                continue;
            }
            popped_degree++;
            if (!vt.set(v)) {
                continue;
            }
            unvisited_count++;
            const float dis = qdis(v);
            result.distance_count++;
            if (top_candidates.size() < ef_search || dis < lower_bound) {
                candidate_set.emplace(dis, v);
                accepted_count++;
                if (!sel || sel->is_member(v)) {
                    top_candidates.emplace(dis, v);
                    if (top_candidates.size() > ef_search) {
                        top_candidates.pop();
                    }
                    if (!top_candidates.empty()) {
                        lower_bound = top_candidates.top().first;
                    }
                }
                if (dis < best_raw_distance) {
                    best_raw_distance = dis;
                }
            }
        }

        const size_t rs_size_after = top_candidates.size();
        const bool is_full = rs_size_after == ef_search;
        float runtime_accepted_rate = nan;
        float runtime_cfr = nan;
        float runtime_smoothed_cfr = nan;
        size_t full_pop_count_after = 0;
        const float furthest_raw_distance = top_candidates.empty() ? nan : top_candidates.top().first;
        const float furthest_dist = sage_external_distance(furthest_raw_distance, similarity_metric);
        const float popped_query_dist = sage_external_distance(candidate_distance, similarity_metric);
        const float best_dist = sage_external_distance(best_raw_distance, similarity_metric);
        const float internal_dist = sage_external_distance(raw_internal_before, similarity_metric);

        if (is_full) {
            full_pop_count++;
            full_pop_count_after = full_pop_count;
            runtime_accepted_rate = unvisited_count > 0
                    ? static_cast<float>(accepted_count) / static_cast<float>(unvisited_count)
                    : 0.0f;
            if (std::isfinite(furthest_dist) && std::fabs(furthest_dist) > 1e-6f) {
                runtime_cfr = popped_query_dist / std::max(furthest_dist, 1e-6f);
                if (std::isnan(smoothed_cfr_ema)) {
                    smoothed_cfr_ema = runtime_cfr;
                } else {
                    smoothed_cfr_ema =
                            cfr_config.cfr_ema_decay * smoothed_cfr_ema +
                            cfr_ema_update * runtime_cfr;
                }
                runtime_smoothed_cfr = smoothed_cfr_ema;
            }
            if (full_pop_count >=
                        static_cast<size_t>(cfr_config.classify_start) &&
                full_pop_count <=
                        static_cast<size_t>(cfr_config.classify_end) &&
                std::isfinite(runtime_smoothed_cfr)) {
                classify_window_sum += static_cast<double>(runtime_smoothed_cfr);
                classify_window_count++;
            }
        }

        if (trace) {
            if (stored_steps < static_cast<size_t>(trace->max_steps)) {
                const size_t pos = static_cast<size_t>(trace_row) * static_cast<size_t>(trace->max_steps) + stored_steps;
                trace->node_labels[pos] = current;
                trace->rs_sizes[pos] = static_cast<uint64_t>(rs_size_before);
                trace->rs_sizes_after[pos] = static_cast<uint64_t>(rs_size_after);
                trace->is_full_pop_after[pos] = is_full ? 1 : 0;
                trace->full_pop_counts_after[pos] = static_cast<uint64_t>(full_pop_count_after);
                trace->popped_degrees[pos] = static_cast<uint64_t>(popped_degree);
                trace->unvisited_counts[pos] = static_cast<uint64_t>(unvisited_count);
                trace->accepted_counts[pos] = static_cast<uint64_t>(accepted_count);
                trace->runtime_accepted_rates[pos] = runtime_accepted_rate;
                trace->runtime_cfrs[pos] = runtime_cfr;
                trace->runtime_smoothed_cfrs[pos] = runtime_smoothed_cfr;
                trace->internal_dists[pos] = internal_dist;
                trace->popped_query_dists[pos] = popped_query_dist;
                trace->furthest_dists[pos] = furthest_dist;
                trace->best_dists[pos] = best_dist;
                trace->top_k_dists[pos] = sage_rank_distance_or_nan(
                        top_candidates, normalized_k, similarity_metric);
                trace->ef_half_dists[pos] = sage_rank_distance_or_nan(
                        top_candidates, std::max<size_t>(1, ef_search / 2), similarity_metric);
                trace->ef_quarter_dists[pos] = sage_rank_distance_or_nan(
                        top_candidates, std::max<size_t>(1, ef_search / 4), similarity_metric);
                trace->sqrt_ef_dists[pos] = sage_rank_distance_or_nan(
                        top_candidates,
                        std::max<size_t>(1, static_cast<size_t>(std::sqrt(static_cast<double>(ef_search)))),
                        similarity_metric);
                trace->top_2k_dists[pos] = sage_rank_distance_or_nan(
                        top_candidates, normalized_k * 2, similarity_metric);
                trace->top_3k_dists[pos] = sage_rank_distance_or_nan(
                        top_candidates, normalized_k * 3, similarity_metric);
                stored_steps++;
            } else {
                truncated = true;
            }
        }
    }

    result.exhausted = candidate_set.empty();
    result.truncated = truncated;
    result.stored_steps = stored_steps;
    result.full_pop_count = full_pop_count;
    result.window_obs_count = classify_window_count;
    if (classify_window_count > 0) {
        result.mean_smoothed_cfr_classify_window = static_cast<float>(
                classify_window_sum / static_cast<double>(classify_window_count));
    }
    result.usable_for_mean_window = full_pop_count >= 16 &&
            std::isfinite(result.mean_smoothed_cfr_classify_window);
    result.closest_dist = sage_external_distance(best_raw_distance, similarity_metric);

    if (out_distances && out_labels) {
        std::vector<HNSW::Node> sorted;
        sorted.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            sorted.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::sort(sorted.begin(), sorted.end(), [](const HNSW::Node& a, const HNSW::Node& b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });
        for (size_t i = 0; i < normalized_k; i++) {
            if (i < sorted.size()) {
                out_distances[i] = sage_external_distance(sorted[i].first, similarity_metric);
                out_labels[i] = sorted[i].second;
            } else {
                out_distances[i] = std::numeric_limits<float>::infinity();
                out_labels[i] = -1;
            }
        }
    }

    return result;
}

struct SageLidSearchResult {
    size_t distance_count = 0;
    size_t hop_count = 0;
    std::vector<float> distances;
};

static SageLidSearchResult sage_search_level0_lid_hnswlib_style(
        const IndexHNSW& index,
        DistanceComputer& qdis,
        VisitedTable& vt,
        HNSW::storage_idx_t entry,
        float entry_distance,
        idx_t ef,
        bool similarity_metric) {
    SageLidSearchResult result;
    const HNSW& hnsw = index.hnsw;
    const size_t ef_search = std::max<size_t>(static_cast<size_t>(ef), 1);

    std::priority_queue<HNSW::Node> top_candidates;
    std::priority_queue<HNSW::Node, std::vector<HNSW::Node>, std::greater<HNSW::Node>>
            candidate_set;

    vt.set(entry);
    top_candidates.emplace(entry_distance, entry);
    candidate_set.emplace(entry_distance, entry);

    while (!candidate_set.empty()) {
        const float candidate_distance = candidate_set.top().first;
        const float lower_bound = top_candidates.empty()
                ? std::numeric_limits<float>::max()
                : top_candidates.top().first;

        // Match hnswlib searchBaseLayerST<true>: bare-bone search stops on
        // candidate_dist > lowerBound, without requiring top_candidates == ef.
        if (candidate_distance > lower_bound) {
            break;
        }

        const HNSW::storage_idx_t current = candidate_set.top().second;
        candidate_set.pop();
        result.hop_count++;

        size_t begin = 0, end = 0;
        hnsw.neighbor_range(current, 0, &begin, &end);
        float mutable_lower_bound = lower_bound;
        for (size_t j = begin; j < end; j++) {
            const HNSW::storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) {
                break;
            }
            if (!vt.set(v)) {
                continue;
            }
            const float dis = qdis(v);
            result.distance_count++;

            if (top_candidates.size() < ef_search || mutable_lower_bound > dis) {
                candidate_set.emplace(dis, v);
                top_candidates.emplace(dis, v);
                while (top_candidates.size() > ef_search) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    mutable_lower_bound = top_candidates.top().first;
                }
            }
        }
    }

    result.distances.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
        result.distances.push_back(
                sage_external_distance(top_candidates.top().first, similarity_metric));
        top_candidates.pop();
    }
    std::sort(result.distances.begin(), result.distances.end());
    return result;
}

static float sage_mle_lid_hnswlib_style(
        const std::vector<float>& sorted_distances,
        idx_t k_lid) {
    if (k_lid < 2) {
        return 0.0f;
    }

    float d_max = 0.0f;
    size_t actual_k = 0;
    for (float distance : sorted_distances) {
        if (!std::isfinite(distance) || distance <= 1e-9f) {
            continue;
        }
        actual_k++;
        d_max = distance;
        if (actual_k >= static_cast<size_t>(k_lid)) {
            break;
        }
    }

    if (actual_k < 2 || d_max <= 1e-9f) {
        return 0.0f;
    }

    double sum_log = 0.0;
    size_t valid_k = 0;
    for (float distance : sorted_distances) {
        if (!std::isfinite(distance) || distance <= 1e-9f) {
            continue;
        }
        sum_log += std::log(static_cast<double>(distance / d_max));
        valid_k++;
        if (valid_k >= actual_k) {
            break;
        }
    }

    if (sum_log != 0.0) {
        return static_cast<float>(-static_cast<double>(valid_k) / sum_log);
    }
    return 0.0f;
}


void hnsw_add_vertices(
        IndexHNSW& index_hnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false,
        const size_t* ef_construction_overrides = nullptr) {
    size_t d = index_hnsw.d;
    HNSW& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    auto& locks = index_hnsw.locks;
    locks.prepare(ntotal);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (size_t i = 0; i < n; i++) {
            storage_idx_t pt_id = static_cast<storage_idx_t>(i + n0);
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= static_cast<int>(hist.size())) {
                hist.push_back(0);
            }
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (size_t i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (size_t i = 0; i < n; i++) {
            storage_idx_t pt_id = static_cast<storage_idx_t>(i + n0);
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    size_t check_efConstruction = hnsw.efConstruction > 0
            ? static_cast<size_t>(hnsw.efConstruction)
            : size_t(1);
    if (ef_construction_overrides) {
        for (size_t i = 0; i < n; i++) {
            if (ef_construction_overrides[i] > check_efConstruction) {
                check_efConstruction = ef_construction_overrides[i];
            }
        }
    }
    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hnsw.d * check_efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        size_t i1 = static_cast<int>(n);

        for (int pt_level = static_cast<int>(hist.size()) - 1;
             pt_level >= int(!index_hnsw.init_level0);
             pt_level--) {
            size_t i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %zu elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (size_t j = i0; j < i1; j++) {
                std::swap(
                        order[j],
                        order[j + rng2.rand_int(static_cast<int>(i1 - j))]);
            }

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal, hnsw.use_visited_hashset);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_hnsw.storage));
                bool do_display = verbose && omp_get_thread_num() == 0;
                size_t prev_display = 0;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int64_t i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    int efConstruction = 0;
                    if (ef_construction_overrides) {
                        efConstruction = static_cast<int>(
                                ef_construction_overrides[pt_id - n0]);
                    }

                    hnsw.add_with_locks(
                            *dis,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_hnsw.keep_max_size_level0 && (pt_level == 0),
                            efConstruction);

                    if (do_display && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %zu / %zu\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        if (index_hnsw.init_level0) {
            FAISS_ASSERT(i1 == 0);
        } else {
            FAISS_ASSERT((i1 - hist[0]) == 0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }
    if (!index_hnsw.retain_locks) {
        locks.clear();
    }
}

} // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d_in, int M, MetricType metric)
        : Index(d_in, metric), hnsw(M) {}

IndexHNSW::IndexHNSW(Index* storage_in, int M)
        : Index(storage_in->d, storage_in->metric_type),
          hnsw(M),
          storage(storage_in) {
    metric_arg = storage->metric_arg;
}

IndexHNSW::~IndexHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSW::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

namespace {

template <class BlockResultHandler>
void hnsw_search(
        const IndexHNSW* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    const HNSW& hnsw = index->hnsw;

    int efSearch = hnsw.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            efSearch = hnsw_params->efSearch;
        }
    }
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<VisitedTable> vt;
            std::unique_ptr<typename BlockResultHandler::SingleResultHandler>
                    res;
            std::unique_ptr<DistanceComputer> dis;
            try {
                vt = std::make_unique<VisitedTable>(
                        index->ntotal, hnsw.use_visited_hashset);
                res = std::make_unique<
                        typename BlockResultHandler::SingleResultHandler>(bres);
                dis.reset(storage_distance_computer(index->storage));
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    res->begin(i);
                    dis->set_query(x + i * index->d);

                    HNSWStats stats =
                            hnsw.search(*dis, index, *res, *vt, params);
                    n1 += stats.n1;
                    n2 += stats.n2;
                    ndis += stats.ndis;
                    nhops += stats.nhops;
                    res->end();
                    vt->advance();
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

template <class BlockResultHandler>
void hnsw_search_adaptive_light(
        const IndexHNSW* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params_in) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");

    const HNSW& hnsw = index->hnsw;
    SearchParametersHNSWAdaptiveLight default_params;
    default_params.efSearch = hnsw.efSearch;
    default_params.bounded_queue = true;

    const SearchParametersHNSWAdaptiveLight* params = &default_params;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSWAdaptiveLight*>(
                params_in);
        FAISS_THROW_IF_NOT_MSG(
                params,
                "params must be SearchParametersHNSWAdaptiveLight");
    }

    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

    // adaptive-light query cost is bimodal: easy queries early-stop cheaply
    // while hard queries run to efSearch. Unlike vanilla hnsw_search, we do NOT
    // fragment the query stream into check_period-sized chunks with a fresh
    // parallel region (and implicit barrier) per chunk: with bimodal costs each
    // barrier stalls every thread on the chunk's single hardest query, and
    // because check_period is proportional to 1/efSearch that stall grows as ef
    // rises, collapsing multi-thread QPS at high ef. Instead we open ONE
    // parallel region over the whole query set (schedule(dynamic) hides the
    // hard-query tail across all n queries) and check for interruption inside
    // the loop on a per-thread counter, decoupling interrupt cadence from
    // parallel granularity (same approach as the add path).
    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * std::max(params->efSearch, 1));

    std::exception_ptr ex;
    std::atomic<bool> interrupt{false};

#pragma omp parallel if (n > 1)
    {
        std::unique_ptr<VisitedTable> vt;
        std::unique_ptr<typename BlockResultHandler::SingleResultHandler> res;
        std::unique_ptr<DistanceComputer> dis;
        try {
            vt = std::make_unique<VisitedTable>(
                    index->ntotal, hnsw.use_visited_hashset);
            res = std::make_unique<
                    typename BlockResultHandler::SingleResultHandler>(bres);
            dis.reset(storage_distance_computer(index->storage));
        } catch (...) {
            omp_capture_exception(ex, [&] { interrupt = true; });
        }

        size_t counter = 0;
#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(dynamic)
        for (idx_t i = 0; i < n; i++) {
            if (interrupt.load(std::memory_order_relaxed)) {
                continue;
            }
            try {
                res->begin(i);
                dis->set_query(x + i * index->d);

                HNSWStats stats = hnsw.search_adaptive_light(
                        *dis, index, *res, *vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                nhops += stats.nhops;
                res->end();
                vt->advance();

                if (counter++ % check_period == 0 &&
                    InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }
        }
    }
    omp_rethrow_if_exception(ex);
    InterruptCallback::check();

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

} // anonymous namespace

void IndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    hnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (idx_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::knn_query_adaptive_light(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    hnsw_search_adaptive_light(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        for (idx_t i = 0; i < k * n; i++) {
            if (labels[i] < 0) {
                distances[i] = std::numeric_limits<float>::infinity();
            } else if (std::isfinite(distances[i])) {
                distances[i] = 1.0f + distances[i];
            }
        }
    }
}

void IndexHNSW::knn_query_adaptive_analysis(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        uint64_t* pop_steps,
        uint64_t* stop_flags,
        uint64_t* distance_counts,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(x);
    FAISS_THROW_IF_NOT(distances);
    FAISS_THROW_IF_NOT(labels);
    FAISS_THROW_IF_NOT(pop_steps);
    FAISS_THROW_IF_NOT(stop_flags);
    FAISS_THROW_IF_NOT(distance_counts);

    const HNSW& hnsw = this->hnsw;
    SearchParametersHNSWAdaptiveLight default_params;
    default_params.efSearch = hnsw.efSearch;
    default_params.bounded_queue = true;

    const SearchParametersHNSWAdaptiveLight* params = &default_params;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSWAdaptiveLight*>(
                params_in);
        FAISS_THROW_IF_NOT_MSG(
                params,
                "params must be SearchParametersHNSWAdaptiveLight");
    }

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;
    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * this->d * std::max(params->efSearch, 1));

    std::exception_ptr ex;
    std::atomic<bool> interrupt{false};

#pragma omp parallel if (n > 1)
    {
        std::unique_ptr<VisitedTable> vt;
        std::unique_ptr<RH::SingleResultHandler> res;
        std::unique_ptr<DistanceComputer> dis;
        try {
            vt = std::make_unique<VisitedTable>(
                    this->ntotal, hnsw.use_visited_hashset);
            res = std::make_unique<RH::SingleResultHandler>(bres);
            dis.reset(storage_distance_computer(this->storage));
        } catch (...) {
            omp_capture_exception(ex, [&] { interrupt = true; });
        }

        size_t counter = 0;
#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(dynamic)
        for (idx_t i = 0; i < n; i++) {
            if (interrupt.load(std::memory_order_relaxed)) {
                continue;
            }
            try {
                res->begin(i);
                dis->set_query(x + i * this->d);

                HNSWStats stats = hnsw.search_adaptive_light(
                        *dis, this, *res, *vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                nhops += stats.nhops;
                pop_steps[i] = static_cast<uint64_t>(stats.nhops);
                stop_flags[i] = 0;
                distance_counts[i] = static_cast<uint64_t>(stats.ndis);
                res->end();
                vt->advance();

                if (counter++ % check_period == 0 &&
                    InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }
        }
    }
    omp_rethrow_if_exception(ex);
    InterruptCallback::check();

    HNSWStats total;
    total.n1 = n1;
    total.n2 = n2;
    total.ndis = ndis;
    total.nhops = nhops;
    hnsw_stats.combine(total);

    if (is_similarity_metric(this->metric_type)) {
        for (idx_t i = 0; i < k * n; i++) {
            if (labels[i] < 0) {
                distances[i] = std::numeric_limits<float>::infinity();
            } else if (std::isfinite(distances[i])) {
                distances[i] = 1.0f + distances[i];
            }
        }
    }
}

void IndexHNSW::knn_query_hide_node(
        idx_t n,
        const float* x,
        const idx_t* hide_labels,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(x);
    FAISS_THROW_IF_NOT(hide_labels);
    FAISS_THROW_IF_NOT(distances);
    FAISS_THROW_IF_NOT(labels);

    const bool similarity_metric = is_similarity_metric(this->metric_type);
    const HNSW& hnsw = this->hnsw;
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;
    idx_t check_period = InterruptCallback::get_period_hint(
            std::max<int>(hnsw.max_level, 1) * this->d * std::max<idx_t>(k, 1));

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<VisitedTable> vt;
            std::unique_ptr<DistanceComputer> dis;
            try {
                vt = std::make_unique<VisitedTable>(
                        this->ntotal, hnsw.use_visited_hashset);
                dis.reset(storage_distance_computer(this->storage));
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

            // bimodal per-query cost (see knn_query_adaptive_light): dynamic
            // avoids the straggler imbalance that guided causes here.
#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(dynamic)
            for (idx_t i = i0; i < i1; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    float* row_distances = distances + i * k;
                    idx_t* row_labels = labels + i * k;
                    for (idx_t j = 0; j < k; j++) {
                        row_distances[j] = std::numeric_limits<float>::infinity();
                        row_labels[j] = -1;
                    }

                    dis->set_query(x + i * this->d);
                    SageEntryPointResult entry = sage_resolve_search_entry_hidden(
                            *this, *dis, hide_labels[i]);
                    if (entry.nearest >= 0) {
                        SageLevel0SearchResult stats = sage_search_level0_hidden(
                                *this,
                                *dis,
                                *vt,
                                entry.nearest,
                                entry.distance,
                                k,
                                std::max<idx_t>(hnsw.efSearch, k),
                                hide_labels[i],
                                similarity_metric,
                                params,
                                row_distances,
                                row_labels,
                                nullptr,
                                i);
                        n1 += 1;
                        n2 += stats.exhausted ? 1 : 0;
                        ndis += entry.distance_count + stats.distance_count;
                        nhops += stats.hop_count;
                    }
                    vt->advance();
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

void IndexHNSW::compute_internal_lids(
        idx_t n,
        const idx_t* ids,
        idx_t k_lid,
        float* lids,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(k_lid > 0);
    FAISS_THROW_IF_NOT(ids);
    FAISS_THROW_IF_NOT(lids);
    (void)params;

    const bool similarity_metric = is_similarity_metric(this->metric_type);
    const HNSW& hnsw = this->hnsw;
    if (this->ntotal <= 0 || hnsw.entry_point < 0) {
        for (idx_t i = 0; i < n; i++) {
            lids[i] = 0.0f;
        }
        return;
    }

    // Match hnswlib calcNodeLidValueInternal(): searchBaseLayerST<true>(
    // enterpoint_node_, query_data, max(default ef_=10, k_lid + 1)).  FAISS
    // DARTH indexes often persist efSearch=1000, so do not reuse hnsw.efSearch
    // here; otherwise offline LID cost and buckets no longer correspond to the
    // hnswlib SAGE pipeline.
    const idx_t lid_ef = std::max<idx_t>(10, k_lid + 1);

    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;
    idx_t check_period = InterruptCallback::get_period_hint(
            std::max<int>(hnsw.max_level, 1) * this->d * std::max<idx_t>(lid_ef, 1));

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<VisitedTable> vt;
            std::unique_ptr<DistanceComputer> dis;
            std::vector<float> query(this->d);
            try {
                vt = std::make_unique<VisitedTable>(
                        this->ntotal, hnsw.use_visited_hashset);
                dis.reset(storage_distance_computer(this->storage));
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    const idx_t query_id = ids[i];
                    lids[i] = 0.0f;
                    if (query_id < 0 || query_id >= this->ntotal) {
                        continue;
                    }

                    storage->reconstruct(query_id, query.data());
                    dis->set_query(query.data());
                    const HNSW::storage_idx_t entry = hnsw.entry_point;
                    const float entry_distance = (*dis)(entry);
                    SageLidSearchResult stats = sage_search_level0_lid_hnswlib_style(
                            *this,
                            *dis,
                            *vt,
                            entry,
                            entry_distance,
                            lid_ef,
                            similarity_metric);

                    n1 += 1;
                    ndis += 1 + stats.distance_count;
                    nhops += stats.hop_count;
                    lids[i] = sage_mle_lid_hnswlib_style(stats.distances, k_lid);
                    vt->advance();
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

void IndexHNSW::search_layer0_trace(
        idx_t n,
        const float* x,
        idx_t k,
        idx_t ef,
        const idx_t* hide_labels,
        idx_t max_steps,
        uint64_t* step_counts,
        uint64_t* truncated_flags,
        uint64_t* distance_counts,
        float* closest_dists,
        idx_t* node_labels,
        uint64_t* rs_sizes,
        uint64_t* rs_sizes_after,
        uint64_t* is_full_pop_after,
        uint64_t* full_pop_counts_after,
        uint64_t* popped_degrees,
        uint64_t* unvisited_counts,
        uint64_t* accepted_counts,
        float* runtime_accepted_rates,
        float* runtime_cfrs,
        float* runtime_smoothed_cfrs,
        float* internal_dists,
        float* popped_query_dists,
        float* furthest_dists,
        float* best_dists,
        float* top_k_dists,
        float* ef_half_dists,
        float* ef_quarter_dists,
        float* sqrt_ef_dists,
        float* top_2k_dists,
        float* top_3k_dists,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(ef > 0);
    FAISS_THROW_IF_NOT(max_steps > 0);
    FAISS_THROW_IF_NOT(x);
    FAISS_THROW_IF_NOT(hide_labels);
    FAISS_THROW_IF_NOT(step_counts);
    FAISS_THROW_IF_NOT(truncated_flags);
    FAISS_THROW_IF_NOT(distance_counts);
    FAISS_THROW_IF_NOT(closest_dists);
    FAISS_THROW_IF_NOT(node_labels);
    FAISS_THROW_IF_NOT(rs_sizes);
    FAISS_THROW_IF_NOT(rs_sizes_after);
    FAISS_THROW_IF_NOT(is_full_pop_after);
    FAISS_THROW_IF_NOT(full_pop_counts_after);
    FAISS_THROW_IF_NOT(popped_degrees);
    FAISS_THROW_IF_NOT(unvisited_counts);
    FAISS_THROW_IF_NOT(accepted_counts);
    FAISS_THROW_IF_NOT(runtime_accepted_rates);
    FAISS_THROW_IF_NOT(runtime_cfrs);
    FAISS_THROW_IF_NOT(runtime_smoothed_cfrs);
    FAISS_THROW_IF_NOT(internal_dists);
    FAISS_THROW_IF_NOT(popped_query_dists);
    FAISS_THROW_IF_NOT(furthest_dists);
    FAISS_THROW_IF_NOT(best_dists);
    FAISS_THROW_IF_NOT(top_k_dists);
    FAISS_THROW_IF_NOT(ef_half_dists);
    FAISS_THROW_IF_NOT(ef_quarter_dists);
    FAISS_THROW_IF_NOT(sqrt_ef_dists);
    FAISS_THROW_IF_NOT(top_2k_dists);
    FAISS_THROW_IF_NOT(top_3k_dists);

    const bool similarity_metric = is_similarity_metric(this->metric_type);
    const HNSW& hnsw = this->hnsw;
    SageTraceOutputs trace;
    trace.max_steps = max_steps;
    trace.step_counts = step_counts;
    trace.truncated_flags = truncated_flags;
    trace.distance_counts = distance_counts;
    trace.closest_dists = closest_dists;
    trace.node_labels = node_labels;
    trace.rs_sizes = rs_sizes;
    trace.rs_sizes_after = rs_sizes_after;
    trace.is_full_pop_after = is_full_pop_after;
    trace.full_pop_counts_after = full_pop_counts_after;
    trace.popped_degrees = popped_degrees;
    trace.unvisited_counts = unvisited_counts;
    trace.accepted_counts = accepted_counts;
    trace.runtime_accepted_rates = runtime_accepted_rates;
    trace.runtime_cfrs = runtime_cfrs;
    trace.runtime_smoothed_cfrs = runtime_smoothed_cfrs;
    trace.internal_dists = internal_dists;
    trace.popped_query_dists = popped_query_dists;
    trace.furthest_dists = furthest_dists;
    trace.best_dists = best_dists;
    trace.top_k_dists = top_k_dists;
    trace.ef_half_dists = ef_half_dists;
    trace.ef_quarter_dists = ef_quarter_dists;
    trace.sqrt_ef_dists = sqrt_ef_dists;
    trace.top_2k_dists = top_2k_dists;
    trace.top_3k_dists = top_3k_dists;

    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;
    idx_t check_period = InterruptCallback::get_period_hint(
            std::max<int>(hnsw.max_level, 1) * this->d * std::max<idx_t>(ef, 1));

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<VisitedTable> vt;
            std::unique_ptr<DistanceComputer> dis;
            try {
                vt = std::make_unique<VisitedTable>(
                        this->ntotal, hnsw.use_visited_hashset);
                dis.reset(storage_distance_computer(this->storage));
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    step_counts[i] = 0;
                    truncated_flags[i] = 0;
                    distance_counts[i] = 0;
                    closest_dists[i] = std::numeric_limits<float>::infinity();

                    dis->set_query(x + i * this->d);
                    SageEntryPointResult entry = sage_resolve_search_entry_hidden(
                            *this, *dis, hide_labels[i]);
                    if (entry.nearest >= 0) {
                        SageLevel0SearchResult stats = sage_search_level0_hidden(
                                *this,
                                *dis,
                                *vt,
                                entry.nearest,
                                entry.distance,
                                k,
                                ef,
                                hide_labels[i],
                                similarity_metric,
                                params,
                                nullptr,
                                nullptr,
                                &trace,
                                i);
                        step_counts[i] = static_cast<uint64_t>(stats.stored_steps);
                        truncated_flags[i] = stats.truncated ? 1 : 0;
                        distance_counts[i] = static_cast<uint64_t>(
                                entry.distance_count + stats.distance_count);
                        closest_dists[i] = stats.closest_dist;
                        n1 += 1;
                        n2 += stats.exhausted ? 1 : 0;
                        ndis += entry.distance_count + stats.distance_count;
                        nhops += stats.hop_count;
                    }
                    vt->advance();
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

void IndexHNSW::search_layer0_cfr_summary(
        idx_t n,
        const float* x,
        idx_t k,
        idx_t ef,
        const idx_t* hide_labels,
        uint64_t* full_pop_counts,
        uint64_t* window_obs_counts,
        uint64_t* usable_flags,
        uint64_t* distance_counts,
        float* mean_smoothed_cfrs,
        float* closest_dists,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(ef > 0);
    FAISS_THROW_IF_NOT(x);
    FAISS_THROW_IF_NOT(hide_labels);
    FAISS_THROW_IF_NOT(full_pop_counts);
    FAISS_THROW_IF_NOT(window_obs_counts);
    FAISS_THROW_IF_NOT(usable_flags);
    FAISS_THROW_IF_NOT(distance_counts);
    FAISS_THROW_IF_NOT(mean_smoothed_cfrs);
    FAISS_THROW_IF_NOT(closest_dists);

    const bool similarity_metric = is_similarity_metric(this->metric_type);
    const HNSW& hnsw = this->hnsw;
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;
    idx_t check_period = InterruptCallback::get_period_hint(
            std::max<int>(hnsw.max_level, 1) * this->d * std::max<idx_t>(ef, 1));

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<VisitedTable> vt;
            std::unique_ptr<DistanceComputer> dis;
            try {
                vt = std::make_unique<VisitedTable>(
                        this->ntotal, hnsw.use_visited_hashset);
                dis.reset(storage_distance_computer(this->storage));
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    full_pop_counts[i] = 0;
                    window_obs_counts[i] = 0;
                    usable_flags[i] = 0;
                    distance_counts[i] = 0;
                    mean_smoothed_cfrs[i] = std::numeric_limits<float>::quiet_NaN();
                    closest_dists[i] = std::numeric_limits<float>::infinity();

                    dis->set_query(x + i * this->d);
                    SageEntryPointResult entry = sage_resolve_search_entry_hidden(
                            *this, *dis, hide_labels[i]);
                    if (entry.nearest >= 0) {
                        SageLevel0SearchResult stats = sage_search_level0_hidden(
                                *this,
                                *dis,
                                *vt,
                                entry.nearest,
                                entry.distance,
                                k,
                                ef,
                                hide_labels[i],
                                similarity_metric,
                                params,
                                nullptr,
                                nullptr,
                                nullptr,
                                i);
                        full_pop_counts[i] = static_cast<uint64_t>(stats.full_pop_count);
                        window_obs_counts[i] = static_cast<uint64_t>(stats.window_obs_count);
                        usable_flags[i] = stats.usable_for_mean_window ? 1 : 0;
                        distance_counts[i] = static_cast<uint64_t>(
                                entry.distance_count + stats.distance_count);
                        mean_smoothed_cfrs[i] = stats.mean_smoothed_cfr_classify_window;
                        closest_dists[i] = stats.closest_dist;
                        n1 += 1;
                        n2 += stats.exhausted ? 1 : 0;
                        ndis += entry.distance_count + stats.distance_count;
                        nhops += stats.hop_count;
                    }
                    vt->advance();
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

void IndexHNSW::knn_query_beam_width_first_target_hit_step(
        idx_t n,
        const float* x,
        idx_t target_k,
        const idx_t* target_labels,
        const uint64_t* target_hits,
        idx_t k,
        idx_t ef_before,
        idx_t switch_pop,
        idx_t switch_full_pop,
        idx_t ef_after,
        uint64_t* first_steps,
        uint64_t* reached_flags,
        uint64_t* achieved_hits,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(target_k > 0);
    FAISS_THROW_IF_NOT(target_labels);
    FAISS_THROW_IF_NOT(target_hits);
    FAISS_THROW_IF_NOT(first_steps);
    FAISS_THROW_IF_NOT(reached_flags);
    FAISS_THROW_IF_NOT(achieved_hits);

    const HNSW& hnsw = this->hnsw;
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;
    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * this->d * std::max<idx_t>(ef_before, 1));

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<VisitedTable> vt;
            std::unique_ptr<DistanceComputer> dis;
            try {
                vt = std::make_unique<VisitedTable>(
                        this->ntotal, hnsw.use_visited_hashset);
                dis.reset(storage_distance_computer(this->storage));
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    dis->set_query(x + i * this->d);
                    HNSWTargetHitStats stats =
                            hnsw.search_first_target_hit_step(
                                    *dis,
                                    this,
                                    *vt,
                                    k,
                                    ef_before,
                                    switch_pop,
                                    switch_full_pop,
                                    ef_after,
                                    target_labels + i * target_k,
                                    static_cast<size_t>(target_k),
                                    static_cast<size_t>(target_hits[i]),
                                    params);
                    first_steps[i] =
                            static_cast<uint64_t>(stats.first_target_hit_step);
                    reached_flags[i] =
                            static_cast<uint64_t>(stats.reached_target);
                    achieved_hits[i] =
                            static_cast<uint64_t>(stats.achieved_hit_count);
                    n1 += stats.search_stats.n1;
                    n2 += stats.search_stats.n2;
                    ndis += stats.search_stats.ndis;
                    nhops += stats.search_stats.nhops;
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

void IndexHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HNSW::C>;
    RH bres(result, is_similarity_metric(metric_type) ? -radius : radius);

    hnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexHNSW::search1(
        const float* x,
        ResultHandler& handler,
        SearchParameters* params) const {
    SingleQueryBlockResultHandler<HNSW::C, false> bres(handler);
    hnsw_search(this, 1, x, bres, params);
}

void IndexHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    size_t n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices(
            *this,
            n0,
            n,
            x,
            verbose,
            hnsw.levels.size() == static_cast<size_t>(ntotal));
}

void IndexHNSW::add_with_variable_ef_construction(
        idx_t n,
        const float* x,
        const size_t* ef_constructions) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(
            ef_constructions,
            "ef_constructions must point to one positive value per input vector");
    for (idx_t i = 0; i < n; i++) {
        FAISS_THROW_IF_NOT_MSG(
                ef_constructions[i] > 0,
                "ef_construction values must be positive");
        FAISS_THROW_IF_NOT_FMT(
                ef_constructions[i] <=
                        static_cast<size_t>(std::numeric_limits<int>::max()),
                "ef_construction value %zu exceeds int max",
                ef_constructions[i]);
    }

    size_t n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices(
            *this,
            n0,
            n,
            x,
            verbose,
            hnsw.levels.size() == static_cast<size_t>(ntotal),
            ef_constructions);
}

void IndexHNSW::reset() {
    hnsw.reset();
    locks.clear();
    storage->reset();
    ntotal = 0;
}

void IndexHNSW::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

/**************************************************************
 * This section of functions were used during the development of HNSW support.
 * They may be useful in the future but are dormant for now, and thus are not
 * unit tested at the moment.
 * shrink_level_0_neighbors
 * search_level_0
 * init_level_0_from_knngraph
 * init_level_0_from_entry_points
 * reorder_links
 * link_singletons
 **************************************************************/
void IndexHNSW::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0) {
                    break;
                }
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HNSW::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size()) {
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                } else {
                    hnsw.neighbors[j] = -1;
                }
            }
        }
    }
}

void IndexHNSW::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    size_t hnsw_ntotal = hnsw.levels.size();

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    std::exception_ptr ex;
    std::atomic<bool> interrupt{false};
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> qdis;
        HNSWStats search_stats;
        std::unique_ptr<VisitedTable> vt;
        std::unique_ptr<RH::SingleResultHandler> res;
        try {
            qdis.reset(storage_distance_computer(storage));
            vt = std::make_unique<VisitedTable>(
                    hnsw_ntotal, hnsw.use_visited_hashset);
            res = std::make_unique<RH::SingleResultHandler>(bres);
        } catch (...) {
            omp_capture_exception(ex, [&] { interrupt = true; });
        }

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            if (interrupt.load(std::memory_order_relaxed)) {
                continue;
            }
            try {
                res->begin(i);
                qdis->set_query(x + i * d);

                hnsw.search_level_0(
                        *qdis.get(),
                        *res,
                        nprobe,
                        nearest + i * nprobe,
                        nearest_d + i * nprobe,
                        search_type,
                        search_stats,
                        *vt,
                        params);
                res->end();
                vt->advance();
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }
        }
#pragma omp critical
        {
            hnsw_stats.combine(search_stats);
        }
    }
    omp_rethrow_if_exception(ex);
    if (is_similarity_metric(this->metric_type)) {
// we need to revert the negated distances
#pragma omp parallel for
        for (int64_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::init_level_0_from_knngraph(
        int k,
        const float* D,
        const idx_t* I) {
    int dest_size = hnsw.nb_neighbors(0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer* qdis = storage_distance_computer(storage);
        std::vector<float> vec(d);
        storage->reconstruct(i, vec.data());
        qdis->set_query(vec.data());

        std::priority_queue<NodeDistFarther> initial_list;

        for (int j = 0; j < k; j++) {
            int v1 = static_cast<int>(I[i * k + j]);
            if (v1 == i) {
                continue;
            }
            if (v1 < 0) {
                break;
            }
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        HNSW::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size()) {
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            } else {
                hnsw.neighbors[j] = -1;
            }
        }
    }
}

void IndexHNSW::init_level_0_from_entry_points(
        int n,
        const storage_idx_t* points,
        const storage_idx_t* nearests) {
    locks.prepare(ntotal);

#pragma omp parallel
    {
        VisitedTable vt(ntotal, hnsw.use_visited_hashset);

        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));
        std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct(pt_id, vec.data());
            dis->set_query(vec.data());

            hnsw.add_links_starting_from(
                    *dis, pt_id, nearest, (*dis)(nearest), 0, locks, vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    if (!retain_locks) {
        locks.clear();
    }
}

void IndexHNSW::reorder_links() {
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances(M);
        std::vector<size_t> order(M);
        std::vector<storage_idx_t> tmp(M);
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (storage_idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp[j - begin] = nj;
            }

            fvec_argsort(end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }
    }
}

void IndexHNSW::link_singletons() {
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (idx_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0) {
                seen[ni] = true;
            }
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1) {
                n_sing_l1++;
            }
        }
    }

    printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
           n_sing,
           ntotal,
           n_sing_l1);

    std::vector<float> recons(singletons.size() * d);
    for (size_t i = 0; i < singletons.size(); i++) {
        FAISS_ASSERT(false); // not implemented
    }
}

void IndexHNSW::permute_entries(const idx_t* perm) {
    auto flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_storage, "don't know how to permute this index");
    flat_storage->permute_entries(perm);
    hnsw.permute_entries(perm);
}

DistanceComputer* IndexHNSW::get_distance_computer() const {
    return storage->get_distance_computer();
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() {
    is_trained = true;
}

IndexHNSWFlat::IndexHNSWFlat(int d_in, int M, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2) ? new IndexFlatL2(d_in)
                                        : new IndexFlat(d_in, metric),
                  M) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHNSWFlatPanorama implementation
 **************************************************************/

IndexHNSWFlatPanorama::IndexHNSWFlatPanorama()
        : IndexHNSWFlat(), cum_sums(), pano(0, 1, 1), num_panorama_levels(0) {}

IndexHNSWFlatPanorama::IndexHNSWFlatPanorama(
        int d_in,
        int M,
        int num_panorama_levels_in,
        MetricType metric)
        : IndexHNSWFlat(d_in, M, metric),
          cum_sums(),
          pano(d_in * sizeof(float), num_panorama_levels_in, 1),
          num_panorama_levels(num_panorama_levels_in) {
    // For now, we only support L2 distance.
    // Supporting dot product and cosine distance is a trivial addition
    // left for future work.
    FAISS_THROW_IF_NOT(metric == METRIC_L2);

    // Enable Panorama search mode.
    // This is not ideal, but is still more simple than making a subclass of
    // HNSW and overriding the search logic.
    hnsw.is_panorama = true;
}

void IndexHNSWFlatPanorama::add(idx_t n, const float* x) {
    idx_t n0 = ntotal;
    cum_sums.resize((ntotal + n) * (pano.n_levels + 1));
    pano.compute_cumulative_sums(cum_sums.data(), n0, n, x);
    IndexHNSWFlat::add(n, x);
}

void IndexHNSWFlatPanorama::reset() {
    cum_sums.clear();
    IndexHNSWFlat::reset();
}

void IndexHNSWFlatPanorama::permute_entries(const idx_t* perm) {
    std::vector<float> new_cum_sums(ntotal * (pano.n_levels + 1));

    for (idx_t i = 0; i < ntotal; i++) {
        idx_t src = perm[i];
        memcpy(&new_cum_sums[i * (pano.n_levels + 1)],
               &cum_sums[src * (pano.n_levels + 1)],
               (pano.n_levels + 1) * sizeof(float));
    }

    std::swap(cum_sums, new_cum_sums);
    IndexHNSWFlat::permute_entries(perm);
}

/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/

IndexHNSWPQ::IndexHNSWPQ() = default;

IndexHNSWPQ::IndexHNSWPQ(
        int d_in,
        int pq_m,
        int M,
        int pq_nbits,
        MetricType metric)
        : IndexHNSW(new IndexPQ(d_in, pq_m, pq_nbits, metric), M) {
    own_fields = true;
    is_trained = false;
}

void IndexHNSWPQ::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/

IndexHNSWSQ::IndexHNSWSQ(
        int d_in,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexHNSW(new IndexScalarQuantizer(d_in, qtype, metric), M) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexHNSWSQ::IndexHNSWSQ() = default;

/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/

IndexHNSW2Level::IndexHNSW2Level(
        Index* quantizer,
        size_t nlist,
        int m_pq,
        int M)
        : IndexHNSW(new Index2Layer(quantizer, nlist, m_pq), M) {
    own_fields = true;
    is_trained = false;
}

IndexHNSW2Level::IndexHNSW2Level() = default;

namespace {

// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0) {
    int nres = nres_in;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        FAISS_ASSERT(v1 >= 0);
        vt.visited[v1] = vt.visno + 1;
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            if (vt.visited[v1] == vt.visno + 1) {
                // nothing to do
            } else {
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.visited[v1] < vt.visno) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
                vt.visited[v1] = vt.visno + 1;
            }
        }

        nstep++;
        if (nstep > hnsw.efSearch) {
            break;
        }
    }

    stats.n1++;
    if (candidates.size() == 0) {
        stats.n2++;
    }

    return nres;
}

} // namespace

void IndexHNSW2Level::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search(n, x, k, distances, labels);

    } else { // "mixed" search
        size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

        const IndexIVFPQ* index_ivfpq =
                dynamic_cast<const IndexIVFPQ*>(storage);
        FAISS_THROW_IF_NOT_MSG(
                index_ivfpq,
                "IndexHNSW2Level mixed search requires IndexIVFPQ storage");

        size_t nprobe = index_ivfpq->nprobe;

        std::unique_ptr<idx_t[]> coarse_assign(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        index_ivfpq->quantizer->search(
                n, x, nprobe, coarse_dis.get(), coarse_assign.get());

        index_ivfpq->search_preassigned(
                n,
                x,
                k,
                coarse_assign.get(),
                coarse_dis.get(),
                distances,
                labels,
                false);

        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};
#pragma omp parallel
        {
            // visited table (not hash set) for tri-state flags.
            std::unique_ptr<VisitedTable> vt;
            std::unique_ptr<DistanceComputer> dis;
            constexpr int candidates_size = 1;
            std::unique_ptr<MinimaxHeap> candidates;
            try {
                vt = std::make_unique<VisitedTable>(
                        ntotal, /*use_hashset=*/false);
                dis.reset(storage_distance_computer(storage));
                candidates = std::make_unique<MinimaxHeap>(candidates_size);
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

#pragma omp for reduction(+ : n1, n2, ndis, nhops)
            for (idx_t i = 0; i < n; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    idx_t* idxi = labels + i * k;
                    float* simi = distances + i * k;
                    dis->set_query(x + i * d);

                    // mark all inverted list elements as visited
                    for (size_t j = 0; j < nprobe; j++) {
                        idx_t key = coarse_assign[j + i * nprobe];
                        if (key < 0) {
                            break;
                        }
                        size_t list_length = index_ivfpq->get_list_size(key);
                        const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                        for (size_t jj = 0; jj < list_length; jj++) {
                            vt->set(ids[jj]);
                        }
                    }

                    candidates->clear();

                    for (int j = 0; j < k; j++) {
                        if (idxi[j] < 0) {
                            break;
                        }
                        candidates->push(
                                static_cast<storage_idx_t>(idxi[j]), simi[j]);
                    }

                    // reorder from sorted to heap
                    maxheap_heapify(k, simi, idxi, simi, idxi, k);

                    HNSWStats search_stats;
                    search_from_candidates_2(
                            hnsw,
                            *dis,
                            k,
                            idxi,
                            simi,
                            *candidates,
                            *vt,
                            search_stats,
                            0,
                            k);
                    n1 += search_stats.n1;
                    n2 += search_stats.n2;
                    ndis += search_stats.ndis;
                    nhops += search_stats.nhops;

                    vt->advance();
                    vt->advance();

                    maxheap_reorder(k, simi, idxi);
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);

        hnsw_stats.combine({n1, n2, ndis, nhops});
    }
}

void IndexHNSW2Level::flip_to_ivf() {
    Index2Layer* storage2l = dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT(storage2l);

    IndexIVFPQ* index_ivfpq = new IndexIVFPQ(
            storage2l->q1.quantizer,
            d,
            storage2l->q1.nlist,
            storage2l->pq.M,
            8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map(true);

    storage = index_ivfpq;
    delete storage2l;
}

/**************************************************************
 * IndexHNSWCagra implementation
 **************************************************************/

IndexHNSWCagra::IndexHNSWCagra() {
    is_trained = true;
}

IndexHNSWCagra::IndexHNSWCagra(
        int d_in,
        int M,
        MetricType metric,
        NumericType numeric_type)
        : IndexHNSW(d_in, M, metric) {
    FAISS_THROW_IF_NOT_MSG(
            ((metric == METRIC_L2) || (metric == METRIC_INNER_PRODUCT)),
            "unsupported metric type for IndexHNSWCagra");
    numeric_type_ = numeric_type;
    if (numeric_type == NumericType::Float32) {
        // Use flat storage with full precision for fp32
        storage = (metric == METRIC_L2)
                ? static_cast<Index*>(new IndexFlatL2(d))
                : static_cast<Index*>(new IndexFlatIP(d));
    } else if (numeric_type == NumericType::Float16) {
        auto qtype = ScalarQuantizer::QT_fp16;
        storage = new IndexScalarQuantizer(d, qtype, metric);
    } else {
        FAISS_THROW_MSG(
                "Unsupported numeric_type: only F16 and F32 are supported for IndexHNSWCagra");
    }

    metric_arg = storage->metric_arg;

    own_fields = true;
    is_trained = true;
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexHNSWCagra::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            !base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexHNSW::add(n, x);
}

void IndexHNSWCagra::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!base_level_only) {
        IndexHNSW::search(n, x, k, distances, labels, params);
    } else {
        if (ntotal == 0) {
            std::fill(
                    distances,
                    distances + n * k,
                    std::numeric_limits<float>::max());
            std::fill(labels, labels + n * k, -1);
            return;
        }
        std::vector<storage_idx_t> nearest(n);
        std::vector<float> nearest_d(n);

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(this->storage));
            dis->set_query(x + i * d);
            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<float>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                auto distance = (*dis)(idx);
                if (distance < nearest_d[i]) {
                    nearest[i] = static_cast<storage_idx_t>(idx);
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

        search_level_0(
                n,
                x,
                k,
                nearest.data(),
                nearest_d.data(),
                distances,
                labels,
                1, // n_probes
                1, // search_type
                params);
    }
}

void IndexHNSWCagra::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    if (!base_level_only) {
        IndexHNSW::range_search(n, x, radius, result, params);
        return;
    }

    const HNSW& hnsw = this->hnsw;
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;
    float threshold = is_similarity_metric(metric_type) ? -radius : radius;
    RangeSearchPartialResult pres(result);

    for (idx_t i = 0; i < n; i++) {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));
        dis->set_query(x + i * d);

        storage_idx_t nearest = -1;
        float nearest_d = std::numeric_limits<float>::max();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<idx_t> distrib(0, ntotal - 1);

        for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
            auto idx = distrib(gen);
            auto distance = (*dis)(idx);
            if (distance < nearest_d) {
                nearest = idx;
                nearest_d = distance;
            }
        }
        FAISS_THROW_IF_NOT_MSG(
                nearest >= 0, "Could not find a valid entrypoint.");

        RangeQueryResult& qres = pres.new_result(i);
        RangeResultHandler<HNSW::C> res(&qres, threshold);
        VisitedTable vt(ntotal, hnsw.use_visited_hashset);
        HNSWStats stats;
        hnsw.search_level_0(
                *dis, res, 1, &nearest, &nearest_d, 1, stats, vt, params);
        n1 += stats.n1;
        n2 += stats.n2;
        ndis += stats.ndis;
        nhops += stats.nhops;
    }

    pres.set_lims();
    result->do_allocation();
    pres.copy_result();

    hnsw_stats.combine({n1, n2, ndis, nhops});

    if (is_similarity_metric(metric_type)) {
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

faiss::NumericType IndexHNSWCagra::get_numeric_type() const {
    return numeric_type_;
}

void IndexHNSWCagra::set_numeric_type(faiss::NumericType numeric_type) {
    numeric_type_ = numeric_type;
}

} // namespace faiss
