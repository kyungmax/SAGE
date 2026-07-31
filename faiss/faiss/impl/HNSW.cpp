/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HNSW.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdlib>
#include <limits>

#include <faiss/IndexHNSW.h>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>

namespace faiss {

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

int HNSW::nb_neighbors(int layer_no) const {
    FAISS_THROW_IF_NOT(
            static_cast<size_t>(layer_no + 1) < cum_nneighbor_per_level.size());
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void HNSW::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (size_t i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int HNSW::cum_nb_neighbors(int layer_no) const {
    FAISS_CHECK_RANGE_DEBUG(layer_no, 0, (int)cum_nneighbor_per_level.size());
    return cum_nneighbor_per_level[layer_no];
}

void HNSW::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    FAISS_CHECK_RANGE_DEBUG(no, 0, (idx_t)offsets.size());
    FAISS_CHECK_RANGE_DEBUG(
            layer_no, 0, (int)cum_nneighbor_per_level.size() - 1);
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

HNSW::HNSW(int M) : rng(12345) {
    set_default_probas(M, 1.0 / log(M));
    offsets.push_back(0);
}

int HNSW::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bisection
    for (size_t level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9) {
            break;
        }
        assign_probas.push_back(proba);
        nn += level == 0 ? M * 2 : M;
        cum_nneighbor_per_level.push_back(nn);
    }
}

void HNSW::clear_neighbor_tables(int level) {
    for (size_t i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = -1;
        }
    }
}

void HNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

void HNSW::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(
            static_cast<size_t>(level) < cum_nneighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
        reduction(+ : tot_reciprocal) reduction(+ : n_node)
    for (idx_t i = 0; i < static_cast<idx_t>(levels.size()); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) {
                    break;
                }
                neighset.insert(neighbors[j]);
            }
            size_t n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0) {
                    break;
                }
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0) {
                        break;
                    }
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %zd\n", n_node);
    printf("   neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

void HNSW::fill_with_random_links(size_t n) {
    int max_level_2 = prepare_level_tab(n);
    RandomGenerator rng2(456);

    for (int level = max_level_2 - 1; level >= 0; --level) {
        std::vector<int> elts;
        for (size_t i = 0; i < n; i++) {
            if (levels[i] > level) {
                elts.push_back(i);
            }
        }
        printf("linking %zd elements in level %d\n", elts.size(), level);

        if (elts.size() == 1) {
            continue;
        }

        for (size_t ii = 0; ii < elts.size(); ii++) {
            int i = elts[ii];
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int other = 0;
                do {
                    other = elts[rng2.rand_int(elts.size())];
                } while (other == i);

                neighbors[j] = other;
            }
        }
    }
}

int HNSW::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (size_t i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int max_level_2 = 0;
    for (size_t i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level_2) {
            max_level_2 = pt_level;
        }
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
    }
    neighbors.resize(offsets.back(), -1);

    return max_level_2;
}

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HNSW::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        size_t max_size,
        bool keep_max_size_level0) {
    // This prevents number of neighbors at
    // level 0 from being shrunk to less than 2 * M.
    // This is essential in making sure
    // `faiss::gpu::GpuIndexCagra::copyFrom(IndexHNSWCagra*)` is functional
    std::vector<NodeDistFarther> outsiders;

    while (input.size() > 0) {
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            if (dist_v1_v2 < dist_v1_q) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(v1);
            if (output.size() >= static_cast<size_t>(max_size)) {
                return;
            }
        } else if (keep_max_size_level0) {
            outsiders.push_back(v1);
        }
    }
    size_t idx = 0;
    while (keep_max_size_level0 &&
           (output.size() < static_cast<size_t>(max_size)) &&
           (idx < outsiders.size())) {
        output.push_back(outsiders[idx++]);
    }
}

namespace {

using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        size_t max_size,
        bool keep_max_size_level0 = false) {
    if (resultSet1.size() < static_cast<size_t>(max_size)) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSW::shrink_neighbor_list(
            qdis, resultSet, returnlist, max_size, keep_max_size_level0);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0 = false) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1) {
                break;
            }
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    size_t max_size = end - begin;
    max_size -= max_size * std::clamp(hnsw.prune_headroom, 0.0f, 0.5f);
    shrink_neighbor_list(qdis, resultSet, max_size, keep_max_size_level0);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

} // namespace

/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version,
        int efConstruction) {
    const size_t effective_efConstruction = efConstruction > 0
            ? static_cast<size_t>(efConstruction)
            : static_cast<size_t>(hnsw.efConstruction);
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        if (currEv.d > results.top().d) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        // The reference version is not used, but kept here because:
        // 1. It is easier to switch back if the optimized version has a problem
        // 2. It serves as a starting point for new optimizations
        // 3. It helps understand the code
        // 4. It ensures the reference version is still compilable if the
        // optimized version changes
        // The reference and the optimized versions' results are compared in
        // test_hnsw.cpp
        if (reference_version) {
            // a reference version
            for (size_t i = begin; i < end; i++) {
                storage_idx_t nodeId = hnsw.neighbors[i];
                if (nodeId < 0) {
                    break;
                }
                if (!vt.set(nodeId)) {
                    continue;
                }

                float dis = qdis(nodeId);
                NodeDistFarther evE1(dis, nodeId);

                if (results.size() < effective_efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results.size() > effective_efConstruction) {
                        results.pop();
                    }
                }
            }
        } else {
            // a faster version

            // the following version processes 4 neighbors at a time
            auto update_with_candidate = [&](const storage_idx_t idx,
                                             const float dis) {
                if (results.size() < effective_efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, idx);
                    candidates.emplace(dis, idx);
                    if (results.size() > effective_efConstruction) {
                        results.pop();
                    }
                }
            };

            int n_buffered = 0;
            storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nodeId = hnsw.neighbors[j];
                if (nodeId < 0) {
                    break;
                }
                if (!vt.set(nodeId)) {
                    continue;
                }

                buffered_ids[n_buffered] = nodeId;
                n_buffered += 1;

                if (n_buffered == 4) {
                    float dis[4];
                    qdis.distances_batch_4(
                            buffered_ids[0],
                            buffered_ids[1],
                            buffered_ids[2],
                            buffered_ids[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        update_with_candidate(buffered_ids[id4], dis[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // process leftovers
            for (int icnt = 0; icnt < n_buffered; icnt++) {
                float dis = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], dis);
            }
        }
    }

    vt.advance();
}

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0,
        int efConstruction) {
    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            *this,
            ptdis,
            link_targets,
            nearest,
            d_nearest,
            level,
            vt,
            false,
            efConstruction);

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    ::faiss::shrink_neighbor_list(ptdis, link_targets, M, keep_max_size_level0);

    std::vector<storage_idx_t> neighbors_to_add;
    neighbors_to_add.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0);
        neighbors_to_add.push_back(other_id);
        link_targets.pop();
    }

    locks.unlock(pt_id);
    for (storage_idx_t other_id : neighbors_to_add) {
        locks.lock(other_id);
        add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0);
        locks.unlock(other_id);
    }
    locks.lock(pt_id);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0,
        int efConstruction) {
    storage_idx_t nearest = entry_point;
    if (nearest == -1) { // avoid locking after the first point.
#pragma omp critical
        if (entry_point == -1) { // double-check under lock.
            max_level = pt_level;
            entry_point = pt_id;
            // leave nearest = -1 to trigger early exit after critical block.
        } else {
            // else: Another thread set the entry point.
            nearest = entry_point;
        }
    }

    if (nearest < 0) {
        return;
    }

    locks.lock(pt_id);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    //  greedy search on upper levels
    for (; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
        add_links_starting_from(
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks,
                vt,
                keep_max_size_level0,
                efConstruction);
    }

    locks.unlock(pt_id);

#pragma omp critical
    {
        if (pt_level > max_level) {
            max_level = pt_level;
            entry_point = pt_id;
        }
    }
}

/**************************************************************
 * Searching
 **************************************************************/

using Node = HNSW::Node;
using C = HNSW::C;

/** Helper to extract search parameters from HNSW and SearchParameters */
static inline void extract_search_params(
        const HNSW& hnsw,
        const SearchParameters* params,
        bool& do_dis_check,
        int& efSearch,
        const IDSelector*& sel) {
    // can be overridden by search params
    do_dis_check = hnsw.check_relative_distance;
    efSearch = hnsw.efSearch;
    sel = nullptr;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            do_dis_check = hnsw_params->check_relative_distance;
            efSearch = hnsw_params->efSearch;
        }
        sel = params->sel;
    }
}

static inline float raw_distance_to_external_distance(
        float raw_distance,
        bool is_similarity_metric) {
    // Convert Faiss internal similarity distances back to positive dissimilarity
    // values for ratio-based stopping logic.
    return is_similarity_metric ? (1.0f + raw_distance) : raw_distance;
}

static inline size_t resolve_scaled_shrink_ef(
        size_t configured_ef,
        double scale,
        size_t min_ef) {
    const size_t raw_ef =
            static_cast<size_t>(std::llround((double)configured_ef * scale));
    return std::min(configured_ef, std::max(min_ef, raw_ef));
}

static inline float get_bucket_gamma_ratio(
        const SearchParametersHNSWAdaptiveLight& params,
        size_t index) {
    switch (index) {
        case 0:
            return params.bucket_gamma_ratio_0;
        case 1:
            return params.bucket_gamma_ratio_1;
        case 2:
            return params.bucket_gamma_ratio_2;
        case 3:
            return params.bucket_gamma_ratio_3;
        case 4:
            return params.bucket_gamma_ratio_4;
        case 5:
            return params.bucket_gamma_ratio_5;
        case 6:
            return params.bucket_gamma_ratio_6;
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}

static inline void validate_paper_bucket_routing_config(
        const SearchParametersHNSWAdaptiveLight& params) {
    const int bucket_count = params.paper_bucket_count;
    FAISS_THROW_IF_NOT_MSG(
            bucket_count >= 2 && bucket_count <= 8,
            "paper_bucket_count must be in [2, 8]");

    float prev_gamma = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < bucket_count - 1; i++) {
        const float gamma = get_bucket_gamma_ratio(params, i);
        FAISS_THROW_IF_NOT_MSG(
                std::isfinite(gamma),
                "bucket_gamma_ratios must be finite");
        FAISS_THROW_IF_NOT_MSG(
                gamma >= 0.0f && gamma <= 1.0f,
                "bucket_gamma_ratios must lie in [0, 1]");
        FAISS_THROW_IF_NOT_MSG(
                gamma >= prev_gamma,
                "bucket_gamma_ratios must be monotone nondecreasing");
        prev_gamma = gamma;
    }
}

static inline size_t resolve_paper_bucket_shrink_ef(
        size_t configured_ef,
        size_t k,
        const SearchParametersHNSWAdaptiveLight& params,
        float classify_cfr_ratio) {
    validate_paper_bucket_routing_config(params);

    const size_t bucket_count = static_cast<size_t>(params.paper_bucket_count);
    size_t selected_bucket_index = bucket_count - 1;
    for (size_t i = 0; i + 1 < bucket_count; i++) {
        if (classify_cfr_ratio <= get_bucket_gamma_ratio(params, i)) {
            selected_bucket_index = i;
            break;
        }
    }

    if (selected_bucket_index + 1 >= bucket_count) {
        return configured_ef;
    }

    size_t routed_ef =
            (configured_ef * (selected_bucket_index + 1)) / bucket_count;
    routed_ef = std::max<size_t>(1, routed_ef);
    routed_ef = std::min(configured_ef, routed_ef);
    return std::max(routed_ef, k);
}

static inline void shrink_minimax_heap_capacity(
        MinimaxHeap& candidates,
        size_t new_capacity) {
    const int new_n = static_cast<int>(std::max<size_t>(1, new_capacity));
    if (new_n >= candidates.n) {
        return;
    }

    candidates.n = new_n;
    while (candidates.k > new_n) {
        if (candidates.ids[0] != -1) {
            --candidates.nvalid;
        }
        heap_pop<MinimaxHeap::HC>(
                candidates.k--, candidates.dis.data(), candidates.ids.data());
    }
}


static inline float minimax_heap_rank_boundary(
        const MinimaxHeap& candidates,
        size_t rank) {
    if (candidates.k <= 0) {
        return std::numeric_limits<float>::max();
    }

    std::vector<float> distances;
    distances.reserve(static_cast<size_t>(std::max(candidates.nvalid, 0)));
    for (int i = 0; i < candidates.k; i++) {
        if (candidates.ids[i] != -1) {
            distances.push_back(candidates.dis[i]);
        }
    }
    if (distances.empty()) {
        return std::numeric_limits<float>::max();
    }

    rank = std::max<size_t>(1, rank);
    rank = std::min(rank, distances.size());
    auto nth = distances.begin() + static_cast<std::ptrdiff_t>(rank - 1);
    std::nth_element(distances.begin(), nth, distances.end());
    return *nth;
}

static inline size_t count_target_hits_in_top_candidates(
        const std::priority_queue<Node>& top_candidates,
        size_t k,
        const idx_t* target_labels,
        size_t target_label_count) {
    if (target_labels == nullptr || target_label_count == 0 ||
        top_candidates.empty()) {
        return 0;
    }

    auto snapshot = top_candidates;
    std::vector<Node> sorted_candidates;
    sorted_candidates.reserve(snapshot.size());
    while (!snapshot.empty()) {
        sorted_candidates.push_back(snapshot.top());
        snapshot.pop();
    }
    std::sort(
            sorted_candidates.begin(),
            sorted_candidates.end(),
            [](const Node& lhs, const Node& rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
            });

    const size_t limit = std::min(k, sorted_candidates.size());
    size_t hit_count = 0;
    for (size_t i = 0; i < limit; i++) {
        const idx_t candidate_label = sorted_candidates[i].second;
        for (size_t j = 0; j < target_label_count; j++) {
            if (candidate_label == target_labels[j]) {
                hit_count++;
                break;
            }
        }
    }
    return hit_count;
}

static inline float prune_candidates_to_ef(
        std::priority_queue<Node>& top_candidates,
        size_t ef) {
    while (top_candidates.size() > ef) {
        top_candidates.pop();
    }
    return top_candidates.empty() ? std::numeric_limits<float>::max()
                                  : top_candidates.top().first;
}

/** Do a BFS on the candidates list */
int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    int nres = nres_in;
    int ndis = 0;

    bool do_dis_check;
    int efSearch;
    const IDSelector* sel;
    extract_search_params(hnsw, params, do_dis_check, efSearch, sel);

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            vt.prefetch(v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        threshold = res.threshold;

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (!sel || sel->is_member(idx)) {
                if (dis < threshold) {
                    if (res.add_result(dis, idx)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }
            candidates.push(idx, dis);
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            saved_j[counter] = v1;
            counter += vt.set(v1) ? 1 : 0;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                ndis += 4;

                counter = 0;
            }
        }

        for (int icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);

            ndis += 1;
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}

int search_from_candidates_adaptive_light(
        const HNSW& hnsw,
        const IndexHNSW* index,
        DistanceComputer& qdis,
        ResultHandler& res,
        int k_search,
        HNSW::storage_idx_t nearest,
        float d_nearest,
        VisitedTable& vt,
        HNSWStats& stats,
        const SearchParametersHNSWAdaptiveLight& params) {
    FAISS_THROW_IF_NOT_FMT(
            params.classify_start >= 0,
            "classify_start must be >= 0, got %d",
            params.classify_start);
    FAISS_THROW_IF_NOT_FMT(
            params.classify_end >= params.classify_start &&
                    params.classify_end >= 1,
            "classify_end must be >= max(classify_start, 1), got start=%d end=%d",
            params.classify_start,
            params.classify_end);
    FAISS_THROW_IF_NOT_MSG(
            std::isfinite(params.cfr_ema_decay) &&
                    params.cfr_ema_decay >= 0.0f &&
                    params.cfr_ema_decay <= 1.0f,
            "cfr_ema_decay must be finite and lie in [0, 1]");
    const int classify_start = params.classify_start;
    const int classify_end = params.classify_end;
    const float cfr_ema_decay = params.cfr_ema_decay;
    const float cfr_ema_update = 1.0f - cfr_ema_decay;

    FAISS_THROW_IF_NOT_MSG(
            params.bounded_queue,
            "adaptive-light requires SearchParametersHNSWAdaptiveLight.bounded_queue=true");

    const bool similarity_metric = is_similarity_metric(index->metric_type);
    const IDSelector* sel = params.sel;

    size_t ef_cur = std::max<size_t>(
            std::max<size_t>(static_cast<size_t>(params.efSearch), 1),
            static_cast<size_t>(k_search));
    ef_cur = std::min(
            ef_cur,
            std::max<size_t>(static_cast<size_t>(params.efMax), 1));
    const size_t configured_ef_cur = ef_cur;

    bool is_easy_query = false;
    bool is_super_easy_query = false;
    bool is_mid_easy_query = false;
    bool classification_evaluated = false;
    bool effective_ef_shrink_applied = false;
    int full_pop_count = 0;
    float smoothed_cfr_ema = std::numeric_limits<float>::quiet_NaN();
    float classify_smoothed_cfr_sum = 0.0f;
    int classify_smoothed_cfr_count = 0;
    float classify_cfr_mean = std::numeric_limits<float>::quiet_NaN();
    const bool direct_classifier_threshold_enabled =
            std::isfinite(params.early_stop_ratio);
    if (params.paper_bucket_mode) {
        validate_paper_bucket_routing_config(params);
    }
    const bool super_easy_policy_enabled = direct_classifier_threshold_enabled &&
            !params.paper_bucket_mode &&
            std::isfinite(params.super_easy_gamma_ratio);
    const bool mid_easy_bucket_policy_enabled =
            direct_classifier_threshold_enabled &&
            !params.paper_bucket_mode &&
            std::isfinite(params.mid_easy_upper_gamma_ratio);
    MinimaxHeap candidates(static_cast<int>(configured_ef_cur));
    candidates.push(nearest, d_nearest);
    vt.set(nearest);

    C::T threshold = res.threshold;
    if (!sel || sel->is_member(nearest)) {
        if (d_nearest < threshold && res.add_result(d_nearest, nearest)) {
            threshold = res.threshold;
        }
    }

    int ndis = 0;
    int nstep = 0;

    auto add_to_heap = [&](const storage_idx_t idx, const float dis) {
        if (!sel || sel->is_member(idx)) {
            if (dis < threshold) {
                if (res.add_result(dis, idx)) {
                    threshold = res.threshold;
                }
            }
        }
        candidates.push(idx, dis);
    };

    while (candidates.size() > 0) {
        float candidate_dist = 0.0f;
        storage_idx_t current = candidates.pop_min(&candidate_dist);
        if (current < 0) {
            break;
        }

        if (candidates.count_below(candidate_dist) >=
            static_cast<int>(ef_cur)) {
            break;
        }

        size_t begin, end;
        hnsw.neighbor_range(current, 0, &begin, &end);

        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            vt.prefetch(v1);
            jmax += 1;
        }

        int counter = 0;
        storage_idx_t saved_j[4];

        for (size_t j = begin; j < jmax; j++) {
            storage_idx_t v1 = hnsw.neighbors[j];
            saved_j[counter] = v1;
            counter += vt.set(v1) ? 1 : 0;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (int id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                ndis += 4;
                counter = 0;
            }
        }

        for (int icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);
            ndis += 1;
        }

        if (candidates.k >= static_cast<int>(ef_cur)) {
            full_pop_count++;
            float furthest_dist = raw_distance_to_external_distance(
                    candidates.max(), similarity_metric);
            const float candidate_dist_external =
                    raw_distance_to_external_distance(
                            candidate_dist,
                            similarity_metric);
            const float cfr =
                    candidate_dist_external / std::max(furthest_dist, 1e-6f);
            if (std::isnan(smoothed_cfr_ema)) {
                smoothed_cfr_ema = cfr;
            } else {
                smoothed_cfr_ema =
                        cfr_ema_decay * smoothed_cfr_ema +
                        cfr_ema_update * cfr;
            }

            if (full_pop_count >= classify_start &&
                full_pop_count <= classify_end) {
                classify_smoothed_cfr_sum += smoothed_cfr_ema;
                classify_smoothed_cfr_count++;

                if (!classification_evaluated &&
                    full_pop_count == classify_end) {
                    classification_evaluated = true;
                    if (direct_classifier_threshold_enabled) {
                        classify_cfr_mean = classify_smoothed_cfr_sum /
                                static_cast<float>(classify_smoothed_cfr_count);
                        is_easy_query =
                                classify_cfr_mean <= params.early_stop_ratio;
                        if (is_easy_query) {
                            const float classify_cfr_ratio =
                                    classify_cfr_mean /
                                    std::max(params.early_stop_ratio, 1e-6f);
                            if (super_easy_policy_enabled) {
                                is_super_easy_query =
                                        classify_cfr_ratio <=
                                        params.super_easy_gamma_ratio;
                            }
                            if (mid_easy_bucket_policy_enabled) {
                                is_mid_easy_query =
                                        classify_cfr_ratio <=
                                        params.mid_easy_upper_gamma_ratio;
                            }
                        }
                    }

                    if (!effective_ef_shrink_applied) {
                        size_t shrunk_ef_cur = configured_ef_cur;
                        if (params.paper_bucket_mode) {
                            if (is_easy_query) {
                                const float classify_cfr_ratio =
                                        classify_cfr_mean /
                                        std::max(params.early_stop_ratio, 1e-6f);
                                shrunk_ef_cur = resolve_paper_bucket_shrink_ef(
                                        configured_ef_cur,
                                        static_cast<size_t>(k_search),
                                        params,
                                        classify_cfr_ratio);
                            }
                        } else {
                            const size_t shrink_super_easy_ef =
                                    resolve_scaled_shrink_ef(
                                            configured_ef_cur,
                                            0.25,
                                            128);
                            const size_t shrink_easy_ef =
                                    std::max<size_t>(1, configured_ef_cur / 2);
                            const size_t shrink_mid_easy_ef =
                                    resolve_scaled_shrink_ef(
                                            configured_ef_cur,
                                            0.50,
                                            128);
                            const size_t shrink_edge_easy_ef =
                                    resolve_scaled_shrink_ef(
                                            configured_ef_cur,
                                            0.75,
                                            256);

                            if (super_easy_policy_enabled && is_super_easy_query) {
                                shrunk_ef_cur = shrink_super_easy_ef;
                            } else if (is_easy_query) {
                                if (mid_easy_bucket_policy_enabled) {
                                    shrunk_ef_cur = is_mid_easy_query
                                            ? shrink_mid_easy_ef
                                            : shrink_edge_easy_ef;
                                } else {
                                    shrunk_ef_cur = shrink_easy_ef;
                                }
                            }
                            shrunk_ef_cur = std::max<size_t>(
                                    shrunk_ef_cur,
                                    static_cast<size_t>(k_search));
                        }

                        if (shrunk_ef_cur < ef_cur) {
                            ef_cur = shrunk_ef_cur;
                            shrink_minimax_heap_capacity(candidates, ef_cur);
                            if (candidates.k > 0) {
                                furthest_dist =
                                        raw_distance_to_external_distance(
                                                candidates.max(),
                                                similarity_metric);
                            }
                        }
                        effective_ef_shrink_applied = true;
                    }
                }
            }

        }

        nstep++;
    }

    stats.n1++;
    if (candidates.size() == 0) {
        stats.n2++;
    }
    stats.ndis += ndis;
    stats.nhops += nstep;
    return 0;
}

HNSWTargetHitStats search_from_candidates_first_target_hit_step(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        HNSW::storage_idx_t nearest,
        float d_nearest,
        VisitedTable& vt,
        idx_t k,
        idx_t ef_before,
        idx_t switch_pop,
        idx_t switch_full_pop,
        idx_t ef_after,
        const idx_t* target_labels,
        size_t target_label_count,
        size_t target_hit_count,
        const SearchParameters* params) {
    HNSWTargetHitStats output;
    output.target_hit_count = target_hit_count;
    if (target_hit_count == 0) {
        output.reached_target = 1;
        return output;
    }

    const IDSelector* sel = params ? params->sel : nullptr;
    const size_t normalized_k = std::max<size_t>(static_cast<size_t>(k), 1);
    const size_t normalized_ef_before =
            std::max<size_t>(static_cast<size_t>(ef_before), normalized_k);
    const size_t normalized_ef_after =
            std::max<size_t>(static_cast<size_t>(ef_after), normalized_k);
    size_t ef_cur = normalized_ef_before;
    bool phase_switch_applied = false;
    size_t pop_count = 0;
    size_t full_pop_count = 0;
    int ndis = 0;

    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>>
            candidate_set;

    float lower_bound = std::numeric_limits<float>::max();
    candidate_set.emplace(d_nearest, nearest);
    vt.set(nearest);
    if (!sel || sel->is_member(nearest)) {
        top_candidates.emplace(d_nearest, nearest);
        lower_bound = d_nearest;
    }

    auto push_candidate = [&](idx_t idx, float dis) {
        candidate_set.emplace(dis, idx);
        if (!sel || sel->is_member(idx)) {
            top_candidates.emplace(dis, idx);
            if (top_candidates.size() > ef_cur) {
                top_candidates.pop();
            }
            if (!top_candidates.empty()) {
                lower_bound = top_candidates.top().first;
            }
        }
    };

    while (!candidate_set.empty()) {
        float candidate_dist = candidate_set.top().first;
        HNSW::storage_idx_t current = candidate_set.top().second;

        if (candidate_dist > lower_bound && top_candidates.size() == ef_cur) {
            break;
        }

        candidate_set.pop();
        pop_count++;

        size_t begin, end;
        hnsw.neighbor_range(current, 0, &begin, &end);

        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            vt.prefetch(v1);
            jmax += 1;
        }

        int counter = 0;
        HNSW::storage_idx_t saved_j[4];

        for (size_t j = begin; j < jmax; j++) {
            HNSW::storage_idx_t v1 = hnsw.neighbors[j];
            saved_j[counter] = v1;
            counter += vt.set(v1) ? 1 : 0;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (int id4 = 0; id4 < 4; id4++) {
                    if (top_candidates.size() < ef_cur ||
                        dis[id4] < lower_bound) {
                        push_candidate(saved_j[id4], dis[id4]);
                    }
                }

                ndis += 4;
                counter = 0;
            }
        }

        for (int icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            if (top_candidates.size() < ef_cur || dis < lower_bound) {
                push_candidate(saved_j[icnt], dis);
            }
            ndis += 1;
        }

        if (!phase_switch_applied && switch_pop > 0 &&
            pop_count >= static_cast<size_t>(switch_pop)) {
            phase_switch_applied = true;
            const size_t prev_ef = ef_cur;
            ef_cur = normalized_ef_after;
            if (ef_cur < prev_ef) {
                lower_bound = prune_candidates_to_ef(top_candidates, ef_cur);
            }
        }

        if (top_candidates.size() == ef_cur) {
            full_pop_count++;
            if (!phase_switch_applied && switch_full_pop > 0 &&
                full_pop_count >= static_cast<size_t>(switch_full_pop)) {
                phase_switch_applied = true;
                const size_t prev_ef = ef_cur;
                ef_cur = normalized_ef_after;
                if (ef_cur < prev_ef) {
                    lower_bound =
                            prune_candidates_to_ef(top_candidates, ef_cur);
                }
            }
        }

        const size_t current_hit_count = count_target_hits_in_top_candidates(
                top_candidates,
                normalized_k,
                target_labels,
                target_label_count);
        output.achieved_hit_count =
                std::max(output.achieved_hit_count, current_hit_count);
        if (current_hit_count >= target_hit_count) {
            output.first_target_hit_step = pop_count;
            output.reached_target = 1;
            break;
        }
    }

    if (!output.reached_target) {
        output.achieved_hit_count = std::max(
                output.achieved_hit_count,
                count_target_hits_in_top_candidates(
                        top_candidates,
                        normalized_k,
                        target_labels,
                        target_label_count));
    }

    output.search_stats.n1 = 1;
    if (candidate_set.empty()) {
        output.search_stats.n2 = 1;
    }
    output.search_stats.ndis = ndis;
    output.search_stats.nhops = pop_count;
    return output;
}

int search_from_candidates_panorama(
        const HNSW& hnsw,
        const IndexHNSW* index,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    int nres = nres_in;
    int ndis = 0;

    bool do_dis_check;
    int efSearch;
    const IDSelector* sel;
    extract_search_params(hnsw, params, do_dis_check, efSearch, sel);

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    // Validate the index type so we can access cumulative sums, n_levels, and
    // get the ability to compute partial dot products.
    const auto* panorama_index =
            dynamic_cast<const IndexHNSWFlatPanorama*>(index);
    FAISS_THROW_IF_NOT_MSG(
            panorama_index, "Index must be a IndexHNSWFlatPanorama");
    auto* flat_codes_qdis = dynamic_cast<FlatCodesDistanceComputer*>(&qdis);
    FAISS_THROW_IF_NOT_MSG(
            flat_codes_qdis,
            "DistanceComputer must be a FlatCodesDistanceComputer");

    // Allocate space for the index array and exact distances.
    size_t M = hnsw.nb_neighbors(0);
    std::vector<idx_t> index_array(M);
    std::vector<float> exact_distances(M);

    const float* query = flat_codes_qdis->q;
    std::vector<float> query_cum_sums(panorama_index->pano.n_levels + 1);
    panorama_index->pano.compute_query_cum_sums(query, query_cum_sums.data());
    float query_norm_sq = query_cum_sums[0] * query_cum_sums[0];

    int nstep = 0;
    const size_t d = static_cast<size_t>(panorama_index->d);

    PanoramaStats local_pano_stats;
    local_pano_stats.reset();

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more than ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // Unlike the vanilla HNSW, we already remove (and compact) the visited
        // nodes from the candidates list at this stage. We also remove nodes
        // that are not selected.
        size_t initial_size = 0;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            const float* cum_sums_v1 = panorama_index->get_cum_sum(v1);
            index_array[initial_size] = v1;
            exact_distances[initial_size] =
                    query_norm_sq + cum_sums_v1[0] * cum_sums_v1[0];

            bool is_selected = !sel || sel->is_member(v1);
            initial_size += is_selected && vt.set(v1) ? 1 : 0;
        }

        local_pano_stats.total_dims += initial_size * d;
        size_t batch_size = initial_size;
        size_t curr_panorama_level = 0;
        const size_t num_panorama_levels = panorama_index->pano.n_levels;
        while (curr_panorama_level < num_panorama_levels && batch_size > 0) {
            float query_cum_norm = query_cum_sums[curr_panorama_level + 1];

            size_t start_dim = curr_panorama_level *
                    panorama_index->pano.level_width_floats;
            size_t end_dim = (curr_panorama_level + 1) *
                    panorama_index->pano.level_width_floats;
            end_dim = std::min(end_dim, static_cast<size_t>(panorama_index->d));

            size_t i = 0;
            size_t next_batch_size = 0;
            for (; i + 3 < batch_size; i += 4) {
                idx_t idx_0 = index_array[i];
                idx_t idx_1 = index_array[i + 1];
                idx_t idx_2 = index_array[i + 2];
                idx_t idx_3 = index_array[i + 3];

                float dp[4];
                flat_codes_qdis->partial_dot_product_batch_4(
                        idx_0,
                        idx_1,
                        idx_2,
                        idx_3,
                        dp[0],
                        dp[1],
                        dp[2],
                        dp[3],
                        start_dim,
                        end_dim - start_dim);
                ndis += 4;

                float new_exact_0 = exact_distances[i + 0] - 2 * dp[0];
                float new_exact_1 = exact_distances[i + 1] - 2 * dp[1];
                float new_exact_2 = exact_distances[i + 2] - 2 * dp[2];
                float new_exact_3 = exact_distances[i + 3] - 2 * dp[3];

                float cum_sum_0 = panorama_index->get_cum_sum(
                        idx_0)[curr_panorama_level + 1];
                float cum_sum_1 = panorama_index->get_cum_sum(
                        idx_1)[curr_panorama_level + 1];
                float cum_sum_2 = panorama_index->get_cum_sum(
                        idx_2)[curr_panorama_level + 1];
                float cum_sum_3 = panorama_index->get_cum_sum(
                        idx_3)[curr_panorama_level + 1];

                float cs_bound_0 = 2.0f * cum_sum_0 * query_cum_norm;
                float cs_bound_1 = 2.0f * cum_sum_1 * query_cum_norm;
                float cs_bound_2 = 2.0f * cum_sum_2 * query_cum_norm;
                float cs_bound_3 = 2.0f * cum_sum_3 * query_cum_norm;

                float lower_bound_0 = new_exact_0 - cs_bound_0;
                float lower_bound_1 = new_exact_1 - cs_bound_1;
                float lower_bound_2 = new_exact_2 - cs_bound_2;
                float lower_bound_3 = new_exact_3 - cs_bound_3;

                // The following code is not the most branch friendly (due to
                // the maintenance of the candidate heap), but micro-benchmarks
                // have shown that it is not worth it to write horrible code to
                // squeeze out those cycles.
                if (lower_bound_0 <= threshold) {
                    exact_distances[next_batch_size] = new_exact_0;
                    index_array[next_batch_size] = idx_0;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_0, new_exact_0);
                }
                if (lower_bound_1 <= threshold) {
                    exact_distances[next_batch_size] = new_exact_1;
                    index_array[next_batch_size] = idx_1;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_1, new_exact_1);
                }
                if (lower_bound_2 <= threshold) {
                    exact_distances[next_batch_size] = new_exact_2;
                    index_array[next_batch_size] = idx_2;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_2, new_exact_2);
                }
                if (lower_bound_3 <= threshold) {
                    exact_distances[next_batch_size] = new_exact_3;
                    index_array[next_batch_size] = idx_3;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_3, new_exact_3);
                }
            }

            // Process the remaining candidates.
            for (; i < batch_size; i++) {
                idx_t idx = index_array[i];

                float dp = flat_codes_qdis->partial_dot_product(
                        idx, start_dim, end_dim - start_dim);
                ndis += 1;
                float new_exact = exact_distances[i] - 2.0f * dp;

                float cum_sum = panorama_index->get_cum_sum(
                        idx)[curr_panorama_level + 1];
                float cs_bound = 2.0f * cum_sum * query_cum_norm;
                float lower_bound = new_exact - cs_bound;

                if (lower_bound <= threshold) {
                    exact_distances[next_batch_size] = new_exact;
                    index_array[next_batch_size] = idx;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx, new_exact);
                }
            }

            local_pano_stats.total_dims_scanned +=
                    batch_size * (end_dim - start_dim);
            batch_size = next_batch_size;
            curr_panorama_level++;
        }

        // Add surviving candidates to the result handler.
        for (size_t i = 0; i < batch_size; i++) {
            idx_t idx = index_array[i];
            if (res.add_result(exact_distances[i], idx)) {
                threshold = res.threshold;
                nres += 1;
            }
            candidates.push(idx, exact_distances[i]);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    indexPanorama_stats.add(local_pano_stats);
    return nres;
}

template <typename T, typename Container, typename Compare>
void reservePriorityQueue(
        std::priority_queue<T, Container, Compare>& q,
        std::size_t size) {
    struct Access : std::priority_queue<T, Container, Compare> {
        using std::priority_queue<T, Container, Compare>::c;
    };
    Access access{std::move(q)};
    access.c.reserve(size);
    q = std::move(access);
}

std::priority_queue<HNSW::Node> search_from_candidate_unbounded(
        const HNSW& hnsw,
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats) {
    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    reservePriorityQueue(top_candidates, ef);

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    reservePriorityQueue(candidates, ef);

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            vt->prefetch(v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (top_candidates.top().first > dis ||
                top_candidates.size() < ef) {
                candidates.emplace(dis, idx);
                top_candidates.emplace(dis, idx);

                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            saved_j[counter] = v1;
            counter += vt->set(v1) ? 1 : 0;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                ndis += 4;

                counter = 0;
            }
        }

        for (int icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);

            ndis += 1;
        }

        stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
}

/// greedily update a nearest vector at a given level
HNSWStats greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    HNSWStats stats;

    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        size_t ndis = 0;

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        auto update_with_candidate = [&](const storage_idx_t idx,
                                         const float dis) {
            if (dis < d_nearest) {
                nearest = idx;
                d_nearest = dis;
            }
        };

        int n_buffered = 0;
        storage_idx_t buffered_ids[4];

        for (size_t j = begin; j < end; j++) {
            storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) {
                break;
            }
            ndis += 1;

            buffered_ids[n_buffered] = v;
            n_buffered += 1;

            if (n_buffered == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        buffered_ids[0],
                        buffered_ids[1],
                        buffered_ids[2],
                        buffered_ids[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    update_with_candidate(buffered_ids[id4], dis[id4]);
                }

                n_buffered = 0;
            }
        }

        // process leftovers
        for (int icnt = 0; icnt < n_buffered; icnt++) {
            float dis = qdis(buffered_ids[icnt]);
            update_with_candidate(buffered_ids[icnt], dis);
        }

        // update stats
        stats.ndis += ndis;
        stats.nhops += 1;

        if (nearest == prev_nearest) {
            return stats;
        }
    }
}

namespace {
using Node = HNSW::Node;
using C = HNSW::C;

// just used as a lower bound for the minmaxheap, but it is set for heap search
int extract_k_from_ResultHandler(ResultHandler& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<RH::SingleResultHandler*>(&res)) {
        return hres->k;
    }
    return 1;
}

} // namespace

HNSWStats HNSW::search(
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler& res,
        VisitedTable& vt,
        const SearchParameters* params) const {
    HNSWStats stats;
    if (entry_point == -1) {
        return stats;
    }
    int k = extract_k_from_ResultHandler(res);

    bool bounded_queue = this->search_bounded_queue;
    int cur_efSearch = this->efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            bounded_queue = hnsw_params->bounded_queue;
            cur_efSearch = hnsw_params->efSearch;
        }
    }

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {
        HNSWStats local_stats =
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    int ef = std::max(cur_efSearch, k);
    if (bounded_queue) { // this is the most common branch, for now we only
                         // support Panorama search in this branch
        MinimaxHeap candidates(ef);

        candidates.push(nearest, d_nearest);

        if (!is_panorama) {
            search_from_candidates(
                    *this, qdis, res, candidates, vt, stats, 0, 0, params);
        } else {
            search_from_candidates_panorama(
                    *this,
                    index,
                    qdis,
                    res,
                    candidates,
                    vt,
                    stats,
                    0,
                    0,
                    params);
        }
    } else {
        std::priority_queue<Node> top_candidates =
                search_from_candidate_unbounded(
                        *this, Node(d_nearest, nearest), qdis, ef, &vt, stats);

        while (top_candidates.size() > static_cast<size_t>(k)) {
            top_candidates.pop();
        }

        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            res.add_result(d, label);
            top_candidates.pop();
        }
    }

    vt.advance();

    return stats;
}

HNSWStats HNSW::search_adaptive_light(
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler& res,
        VisitedTable& vt,
        const SearchParametersHNSWAdaptiveLight* params) const {
    HNSWStats stats;
    if (entry_point == -1) {
        return stats;
    }

    SearchParametersHNSWAdaptiveLight default_params;
    default_params.efSearch = this->efSearch;
    default_params.bounded_queue = true;
    const SearchParametersHNSWAdaptiveLight* active_params =
            params ? params : &default_params;

    int k = extract_k_from_ResultHandler(res);

    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {
        HNSWStats local_stats =
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    search_from_candidates_adaptive_light(
            *this,
            index,
            qdis,
            res,
            k,
            nearest,
            d_nearest,
            vt,
            stats,
            *active_params);

    return stats;
}

HNSWTargetHitStats HNSW::search_first_target_hit_step(
        DistanceComputer& qdis,
        const IndexHNSW* index,
        VisitedTable& vt,
        idx_t k,
        idx_t ef_before,
        idx_t switch_pop,
        idx_t switch_full_pop,
        idx_t ef_after,
        const idx_t* target_labels,
        size_t target_label_count,
        size_t target_hit_count,
        const SearchParameters* params) const {
    HNSWTargetHitStats output;
    output.target_hit_count = target_hit_count;
    if (target_hit_count == 0) {
        output.reached_target = 1;
        return output;
    }
    if (entry_point == -1) {
        return output;
    }

    HNSWStats stats;
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {
        HNSWStats local_stats =
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    output = search_from_candidates_first_target_hit_step(
            *this,
            qdis,
            nearest,
            d_nearest,
            vt,
            k,
            ef_before,
            switch_pop,
            switch_full_pop,
            ef_after,
            target_labels,
            target_label_count,
            target_hit_count,
            params);
    output.search_stats.combine(stats);
    vt.advance();
    (void)index;
    return output;
}

void HNSW::search_level_0(
        DistanceComputer& qdis,
        ResultHandler& res,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HNSWStats& search_stats,
        VisitedTable& vt,
        const SearchParameters* params) const {
    const HNSW& hnsw = *this;

    auto cur_efSearch = hnsw.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            cur_efSearch = hnsw_params->efSearch;
        }
    }

    int k = extract_k_from_ResultHandler(res);

    if (search_type == 1) {
        int nres = 0;

        for (idx_t j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0) {
                break;
            }

            if (vt.get(cj)) {
                continue;
            }

            int candidates_size = std::max(cur_efSearch, k);
            MinimaxHeap candidates(candidates_size);

            candidates.push(cj, nearest_d[j]);

            nres = search_from_candidates(
                    hnsw,
                    qdis,
                    res,
                    candidates,
                    vt,
                    search_stats,
                    0,
                    nres,
                    params);
            nres = std::min(nres, candidates_size);
        }
    } else if (search_type == 2) {
        int candidates_size = std::max(cur_efSearch, int(k));
        candidates_size = std::max(candidates_size, int(nprobe));

        MinimaxHeap candidates(candidates_size);
        for (idx_t j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0) {
                break;
            }
            candidates.push(cj, nearest_d[j]);
        }

        search_from_candidates(
                hnsw, qdis, res, candidates, vt, search_stats, 0, 0, params);
    }
}

void HNSW::permute_entries(const idx_t* map) {
    // remap levels
    storage_idx_t ntotal = levels.size();
    std::vector<storage_idx_t> imap(ntotal); // inverse mapping
    // map: new index -> old index
    // imap: old index -> new index
    for (int i = 0; i < ntotal; i++) {
        assert(map[i] >= 0 && map[i] < ntotal);
        imap[map[i]] = i;
    }
    if (entry_point != -1) {
        entry_point = imap[entry_point];
    }
    std::vector<int> new_levels(ntotal);
    std::vector<size_t> new_offsets(ntotal + 1);
    std::vector<storage_idx_t> new_neighbors(neighbors.size());
    size_t no = 0;
    for (int i = 0; i < ntotal; i++) {
        storage_idx_t o = map[i]; // corresponding "old" index
        new_levels[i] = levels[o];
        for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
            storage_idx_t neigh = neighbors[j];
            new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
        }
        new_offsets[i + 1] = no;
    }
    assert(new_offsets[ntotal] == offsets[ntotal]);
    // swap everyone
    std::swap(levels, new_levels);
    std::swap(offsets, new_offsets);
    neighbors = std::move(new_neighbors);
}

} // namespace faiss
