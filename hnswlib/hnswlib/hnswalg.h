#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <limits>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <fstream>

namespace hnswlib {
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    std::vector<float> node_lid_;

    bool allow_replace_deleted_ = false;

    std::mutex deleted_elements_lock;
    std::unordered_set<tableint> deleted_elements;


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = (maxM0_ + 100) * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    struct AdaptiveSearchStats {
        size_t reduced_steps = 0;
        size_t stop_count = 0;
    };

    struct AdaptiveSearchResult {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        AdaptiveSearchStats stats;
        std::vector<SearchStepInfo> path_info;
    };

    struct TargetHitStepStats {
        size_t first_target_hit_step = 0;
        size_t target_hit_count = 0;
        size_t achieved_hit_count = 0;
        size_t reached_target = 0;
    };

    static size_t resolveScaledShrinkEf(
        size_t configured_ef,
        double scale,
        size_t min_ef
    ) {
        const size_t raw_ef = (size_t)std::llround((double)configured_ef * scale);
        return std::min(configured_ef, std::max(min_ef, raw_ef));
    }

    static void validatePaperBucketRoutingConfig(
        size_t paper_bucket_count,
        const std::vector<float>& paper_bucket_gamma_ratios
    ) {
        if (paper_bucket_count < 2 || paper_bucket_count > 8) {
            throw std::invalid_argument("paper_bucket_count must be in [2, 8].");
        }
        if (paper_bucket_gamma_ratios.size() != paper_bucket_count - 1) {
            throw std::invalid_argument(
                "bucket_gamma_ratios must contain exactly paper_bucket_count - 1 entries."
            );
        }
        float prev_gamma = -std::numeric_limits<float>::infinity();
        for (float gamma : paper_bucket_gamma_ratios) {
            if (!std::isfinite(gamma)) {
                throw std::invalid_argument("bucket_gamma_ratios must be finite.");
            }
            if (gamma < 0.0f || gamma > 1.0f) {
                throw std::invalid_argument("bucket_gamma_ratios must lie in [0, 1].");
            }
            if (gamma < prev_gamma) {
                throw std::invalid_argument(
                    "bucket_gamma_ratios must be monotone nondecreasing."
                );
            }
            prev_gamma = gamma;
        }
    }

    static size_t resolvePaperBucketShrinkEf(
        size_t configured_ef,
        size_t k,
        size_t paper_bucket_count,
        float classify_chr_ratio,
        const std::vector<float>& paper_bucket_gamma_ratios
    ) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float g1 = paper_bucket_gamma_ratios.size() > 0 ? paper_bucket_gamma_ratios[0] : nan;
        const float g2 = paper_bucket_gamma_ratios.size() > 1 ? paper_bucket_gamma_ratios[1] : nan;
        const float g3 = paper_bucket_gamma_ratios.size() > 2 ? paper_bucket_gamma_ratios[2] : nan;
        const float g4 = paper_bucket_gamma_ratios.size() > 3 ? paper_bucket_gamma_ratios[3] : nan;
        const float g5 = paper_bucket_gamma_ratios.size() > 4 ? paper_bucket_gamma_ratios[4] : nan;
        const float g6 = paper_bucket_gamma_ratios.size() > 5 ? paper_bucket_gamma_ratios[5] : nan;
        const float g7 = paper_bucket_gamma_ratios.size() > 6 ? paper_bucket_gamma_ratios[6] : nan;

        size_t selected_bucket_index = paper_bucket_count - 1;
        switch (paper_bucket_count) {
            case 2:
                if (classify_chr_ratio <= g1) selected_bucket_index = 0;
                break;
            case 3:
                if (classify_chr_ratio <= g1) selected_bucket_index = 0;
                else if (classify_chr_ratio <= g2) selected_bucket_index = 1;
                break;
            case 4:
                if (classify_chr_ratio <= g1) selected_bucket_index = 0;
                else if (classify_chr_ratio <= g2) selected_bucket_index = 1;
                else if (classify_chr_ratio <= g3) selected_bucket_index = 2;
                break;
            case 5:
                if (classify_chr_ratio <= g1) selected_bucket_index = 0;
                else if (classify_chr_ratio <= g2) selected_bucket_index = 1;
                else if (classify_chr_ratio <= g3) selected_bucket_index = 2;
                else if (classify_chr_ratio <= g4) selected_bucket_index = 3;
                break;
            case 6:
                if (classify_chr_ratio <= g1) selected_bucket_index = 0;
                else if (classify_chr_ratio <= g2) selected_bucket_index = 1;
                else if (classify_chr_ratio <= g3) selected_bucket_index = 2;
                else if (classify_chr_ratio <= g4) selected_bucket_index = 3;
                else if (classify_chr_ratio <= g5) selected_bucket_index = 4;
                break;
            case 7:
                if (classify_chr_ratio <= g1) selected_bucket_index = 0;
                else if (classify_chr_ratio <= g2) selected_bucket_index = 1;
                else if (classify_chr_ratio <= g3) selected_bucket_index = 2;
                else if (classify_chr_ratio <= g4) selected_bucket_index = 3;
                else if (classify_chr_ratio <= g5) selected_bucket_index = 4;
                else if (classify_chr_ratio <= g6) selected_bucket_index = 5;
                break;
            case 8:
                if (classify_chr_ratio <= g1) selected_bucket_index = 0;
                else if (classify_chr_ratio <= g2) selected_bucket_index = 1;
                else if (classify_chr_ratio <= g3) selected_bucket_index = 2;
                else if (classify_chr_ratio <= g4) selected_bucket_index = 3;
                else if (classify_chr_ratio <= g5) selected_bucket_index = 4;
                else if (classify_chr_ratio <= g6) selected_bucket_index = 5;
                else if (classify_chr_ratio <= g7) selected_bucket_index = 6;
                break;
            default:
                throw std::invalid_argument("paper_bucket_count must be in [2, 8].");
        }

        if (selected_bucket_index + 1 >= paper_bucket_count) {
            return configured_ef;
        }
        size_t routed_ef = (configured_ef * (selected_bucket_index + 1)) / paper_bucket_count;
        routed_ef = std::max((size_t)1, routed_ef);
        routed_ef = std::min(configured_ef, routed_ef);
        return std::max(routed_ef, k);
    }

    static float rankDistanceOrNaN(
        const std::vector<std::pair<dist_t, tableint>>& sorted_candidates,
        size_t rank
    ) {
        if (rank == 0 || sorted_candidates.size() < rank) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return (float)sorted_candidates[rank - 1].first;
    }

    size_t computeTopKTargetHitCount(
        const std::priority_queue<std::pair<dist_t, tableint>,
                                  std::vector<std::pair<dist_t, tableint>>,
                                  CompareByFirst>& top_candidates,
        size_t k,
        const labeltype* target_labels,
        size_t target_label_count
    ) const {
        if (target_labels == nullptr || target_label_count == 0 || top_candidates.empty()) {
            return 0;
        }

        auto snapshot = top_candidates;
        std::vector<std::pair<dist_t, tableint>> sorted_candidates;
        sorted_candidates.reserve(snapshot.size());
        while (!snapshot.empty()) {
            sorted_candidates.push_back(snapshot.top());
            snapshot.pop();
        }
        std::sort(
            sorted_candidates.begin(),
            sorted_candidates.end(),
            [](const std::pair<dist_t, tableint>& lhs, const std::pair<dist_t, tableint>& rhs) {
                return lhs.first < rhs.first;
            }
        );

        const size_t limit = std::min(k, sorted_candidates.size());
        size_t hit_count = 0;
        for (size_t i = 0; i < limit; ++i) {
            const labeltype candidate_label = getExternalLabel(sorted_candidates[i].second);
            for (size_t j = 0; j < target_label_count; ++j) {
                if (candidate_label == target_labels[j]) {
                    hit_count++;
                    break;
                }
            }
        }
        return hit_count;
    }

    void fillTraceStepMetrics(
        SearchStepInfo& step,
        const std::priority_queue<std::pair<dist_t, tableint>,
                                  std::vector<std::pair<dist_t, tableint>>,
                                  CompareByFirst>& top_candidates,
        size_t ef,
        size_t k,
        size_t dim
    ) const {
        step.result_set_size_after = top_candidates.size();
        step.furthest_dist = std::numeric_limits<float>::quiet_NaN();
        step.best_dist = std::numeric_limits<float>::quiet_NaN();
        step.top_k_dist = std::numeric_limits<float>::quiet_NaN();
        step.ef_half_dist = std::numeric_limits<float>::quiet_NaN();
        step.ef_quarter_dist = std::numeric_limits<float>::quiet_NaN();
        step.sqrt_ef_dist = std::numeric_limits<float>::quiet_NaN();
        step.shadow_64_dist = std::numeric_limits<float>::quiet_NaN();
        step.shadow_128_dist = std::numeric_limits<float>::quiet_NaN();
        step.shadow_256_dist = std::numeric_limits<float>::quiet_NaN();
        step.shadow_512_dist = std::numeric_limits<float>::quiet_NaN();
        step.top_2k_dist = std::numeric_limits<float>::quiet_NaN();
        step.top_3k_dist = std::numeric_limits<float>::quiet_NaN();
        step.top_k_node_ids.clear();
        step.furthest_vec.clear();

        if (top_candidates.empty()) {
            return;
        }

        auto snapshot = top_candidates;
        std::vector<std::pair<dist_t, tableint>> sorted_candidates;
        sorted_candidates.reserve(snapshot.size());

        while (!snapshot.empty()) {
            sorted_candidates.push_back(snapshot.top());
            snapshot.pop();
        }

        std::sort(
            sorted_candidates.begin(),
            sorted_candidates.end(),
            [](const std::pair<dist_t, tableint>& lhs, const std::pair<dist_t, tableint>& rhs) {
                return lhs.first < rhs.first;
            }
        );

        step.best_dist = (float)sorted_candidates.front().first;
        step.furthest_dist = (float)sorted_candidates.back().first;
        step.top_k_dist = rankDistanceOrNaN(sorted_candidates, k);
        step.ef_half_dist = rankDistanceOrNaN(sorted_candidates, std::max<size_t>(1, ef / 2));
        step.ef_quarter_dist = rankDistanceOrNaN(sorted_candidates, std::max<size_t>(1, ef / 4));
        step.sqrt_ef_dist = rankDistanceOrNaN(sorted_candidates, std::max<size_t>(1, (size_t)std::sqrt((double)ef)));
        step.shadow_64_dist = rankDistanceOrNaN(sorted_candidates, 64);
        step.shadow_128_dist = rankDistanceOrNaN(sorted_candidates, 128);
        step.shadow_256_dist = rankDistanceOrNaN(sorted_candidates, 256);
        step.shadow_512_dist = rankDistanceOrNaN(sorted_candidates, 512);
        step.top_2k_dist = rankDistanceOrNaN(sorted_candidates, k * 2);
        step.top_3k_dist = rankDistanceOrNaN(sorted_candidates, k * 3);

        const size_t top_k_limit = std::min(k, sorted_candidates.size());
        step.top_k_node_ids.reserve(top_k_limit);
        for (size_t i = 0; i < top_k_limit; ++i) {
            step.top_k_node_ids.push_back(sorted_candidates[i].second);
        }

        tableint furthest_id = sorted_candidates.back().second;
        float* vec_ptr = (float*)getDataByInternalId(furthest_id);
        step.furthest_vec.assign(vec_ptr, vec_ptr + dim);
    }


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    static constexpr tableint HIDDEN_NODE_NONE = std::numeric_limits<tableint>::max();

    inline bool isHiddenNode(tableint internal_id, tableint hidden_internal_id) const {
        return hidden_internal_id != HIDDEN_NODE_NONE && internal_id == hidden_internal_id;
    }

    tableint resolveEntryPointForHiddenNode(tableint hidden_internal_id) const {
        if (cur_element_count == 0 || !isHiddenNode(enterpoint_node_, hidden_internal_id)) {
            return enterpoint_node_;
        }
        tableint fallback = HIDDEN_NODE_NONE;
        int fallback_level = -1;
        for (tableint i = 0; i < cur_element_count; ++i) {
            if (isHiddenNode(i, hidden_internal_id) || isMarkedDeleted(i)) {
                continue;
            }
            if (element_levels_[i] > fallback_level) {
                fallback = i;
                fallback_level = element_levels_[i];
                if (fallback_level >= maxlevel_) {
                    break;
                }
            }
        }
        return fallback == HIDDEN_NODE_NONE ? enterpoint_node_ : fallback;
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    float calcNodeLidValueInternal(tableint internal_id, size_t k_lid) {
        if (k_lid < 2) return 0.0f;

        void* query_data = getDataByInternalId(internal_id);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            knn_res = searchBaseLayerST<true>(enterpoint_node_, query_data, std::max(ef_, k_lid + 1));

        std::vector<float> distances;
        while (!knn_res.empty()) {
            distances.push_back((float)knn_res.top().first);
            knn_res.pop();
        }
        std::sort(distances.begin(), distances.end());

        // MLE LID: -k / sum(log(d_i / d_k)).
        float sum_log = 0.0f;
        float d_max = 0;

        size_t actual_k = 0;
        for (size_t i = 1; i < distances.size() && actual_k < k_lid; ++i) {
            if (distances[i] > 1e-9) {
                actual_k++;
                d_max = distances[i];
            }
        }

        if (actual_k < 2 || d_max <= 1e-9) {
            return 0.0f;
        }

        float valid_k = 0;
        for (size_t i = 1; i < distances.size() && valid_k < actual_k; ++i) {
            if (distances[i] > 1e-9) {
                sum_log += std::log(distances[i] / d_max);
                valid_k++;
            }
        }

        if (sum_log != 0) {
            return - (valid_k / sum_log);
        }
        return 0.0f;
    }

    void calcNodeLidInternal(tableint internal_id, size_t k_lid) {
        node_lid_[internal_id] = calcNodeLidValueInternal(internal_id, k_lid);
    }

    std::tuple<std::vector<SearchStepInfo>, size_t, dist_t>
    searchBaseLayerSTWithTrace(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        size_t k,
        tableint hidden_internal_id = HIDDEN_NODE_NONE
    ) const {
        std::vector<SearchStepInfo> path_info;
        size_t dist_count = 0;
        size_t dim = *((size_t *) dist_func_param_);
        size_t full_pop_count = 0;
        static constexpr float CHR_EMA_DECAY = 0.8f;
        static constexpr float CHR_EMA_UPDATE = 1.0f - CHR_EMA_DECAY;
        float runtime_smoothed_chr = std::numeric_limits<float>::quiet_NaN();

        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        if (isHiddenNode(ep_id, hidden_internal_id)) {
            visited_list_pool_->releaseVisitedList(vl);
            return {path_info, 0, std::numeric_limits<dist_t>::infinity()};
        }

        char* ep_data = getDataByInternalId(ep_id);
        dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
        dist_t lowerBound = dist;
        dist_t min_dist = dist;
        dist_count++;

        top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);
        visited_array[ep_id] = visited_array_tag;
        if (hidden_internal_id != HIDDEN_NODE_NONE && hidden_internal_id < cur_element_count) {
            visited_array[hidden_internal_id] = visited_array_tag;
        }

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            if (candidate_dist > lowerBound && top_candidates.size() == ef) {
                break;
            }

            candidate_set.pop();
            tableint current_node_id = current_node_pair.second;

            SearchStepInfo step{};
            step.node_id = current_node_id;
            step.result_set_size = top_candidates.size();
            step.is_full_pop_after = false;
            step.full_pop_count_after = 0;
            step.runtime_chr = std::numeric_limits<float>::quiet_NaN();
            step.runtime_smoothed_chr = std::numeric_limits<float>::quiet_NaN();
            step.runtime_classify_chr_mean = std::numeric_limits<float>::quiet_NaN();
            step.runtime_classification_evaluated = false;
            step.runtime_is_easy_query = false;
            step.runtime_is_super_easy_query = false;
            step.runtime_is_mid_easy_query = false;
            step.runtime_effective_ef = ef;
            step.popped_query_dist = (float)candidate_dist;

            if (!top_candidates.empty()) {
                step.internal_dist = (float)top_candidates.top().first;
            }

            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            size_t unvisited_count = 0;
            size_t accepted_count = 0;
            step.popped_degree = size;

            for (size_t j = 1; j <= size; j++) {
                tableint candidate_id = *(data + j);
                if (visited_array[candidate_id] == visited_array_tag) continue;

                visited_array[candidate_id] = visited_array_tag;
                unvisited_count++;

                char *currObj1 = getDataByInternalId(candidate_id);
                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                dist_count++;

                if (top_candidates.size() < ef || lowerBound > dist) {
                    candidate_set.emplace(-dist, candidate_id);
                    top_candidates.emplace(dist, candidate_id);
                    accepted_count++;
                    if (dist < min_dist) {
                        min_dist = dist;
                    }

                    if (top_candidates.size() > ef)
                        top_candidates.pop();

                    lowerBound = top_candidates.top().first;
                }
            }

            step.unvisited_neighbor_count = unvisited_count;
            step.accepted_neighbor_count = accepted_count;

            if (top_candidates.size() == ef) {
                step.is_full_pop_after = true;
                full_pop_count++;
                step.full_pop_count_after = full_pop_count;

                float furthest_dist = (float)top_candidates.top().first;
                float chr = (float)candidate_dist / std::max(furthest_dist, 1e-6f);
                step.runtime_chr = chr;
                if (std::isnan(runtime_smoothed_chr)) {
                    runtime_smoothed_chr = chr;
                } else {
                    runtime_smoothed_chr =
                        CHR_EMA_DECAY * runtime_smoothed_chr + CHR_EMA_UPDATE * chr;
                }
                step.runtime_smoothed_chr = runtime_smoothed_chr;
            }

            fillTraceStepMetrics(step, top_candidates, ef, k, dim);
            path_info.push_back(std::move(step));
        }

        visited_list_pool_->releaseVisitedList(vl);
        return {path_info, dist_count, min_dist};
    }

    std::tuple<std::vector<SearchStepInfo>, size_t, dist_t>
    searchKnnWithLayer0Trace(
        const void *query_data,
        size_t ef,
        size_t k,
        tableint hidden_internal_id = HIDDEN_NODE_NONE
    ) const {
        size_t total_dist_count = 0;
        if (cur_element_count == 0) {
            return {std::vector<SearchStepInfo>(), 0, std::numeric_limits<dist_t>::infinity()};
        }

        tableint currObj = resolveEntryPointForHiddenNode(hidden_internal_id);
        if (isHiddenNode(currObj, hidden_internal_id)) {
            return {std::vector<SearchStepInfo>(), 0, std::numeric_limits<dist_t>::infinity()};
        }
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        total_dist_count++;

        int start_level = currObj == enterpoint_node_
            ? maxlevel_
            : std::min(maxlevel_, element_levels_[currObj]);
        for (int level = start_level; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);

                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (isHiddenNode(cand, hidden_internal_id)) {
                        continue;
                    }
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    total_dist_count++;

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        auto [path_info, base_dist_count, closest_dist] =
            searchBaseLayerSTWithTrace(currObj, query_data, ef, k, hidden_internal_id);
        total_dist_count += base_dist_count;

        return {path_info, total_dist_count, closest_dist};
    }

    // Classify early full-pop CHR behavior, then optionally shrink ef once.

template <bool bare_bone_search, bool paper_bucket_mode>
AdaptiveSearchResult
searchBaseLayerAdaptiveAnalysisCore(
    tableint ep_id,
    const void *data_point,
    size_t k,
    size_t ef_init,
    size_t ef_max,
    size_t tmin_pops,
    bool   enable_stop,
    size_t stop_step,
    float  early_stop_ratio,
    BaseFilterFunctor* isIdAllowed,
    float  super_easy_gamma_ratio,
    float  mid_easy_upper_gamma_ratio,
    size_t paper_bucket_count,
    const std::vector<float>& paper_bucket_gamma_ratios,
    int    classify_start = 4,
    int    classify_end = 16,
    float  chr_ema_decay = 0.8f
) const {
    AdaptiveSearchResult output;
    size_t ef_cur = std::max<size_t>(ef_init, k);
    ef_cur = std::min(ef_cur, ef_max);
    const size_t configured_ef_cur = ef_cur;
    size_t dim = *((size_t *) dist_func_param_);

    bool   is_easy_query       = false;
    bool   is_super_easy_query = false;
    bool   is_mid_easy_query   = false;
    bool   classification_evaluated = false;
    bool   effective_ef_shrink_applied = false;
    int    full_pop_count    = 0;
    size_t pop_count         = 0;
    const int   CLASSIFY_START = classify_start;
    const int   CLASSIFY_END = classify_end;
    const float CHR_EMA_DECAY = chr_ema_decay;
    const float CHR_EMA_UPDATE = 1.0f - CHR_EMA_DECAY;
    float smoothed_chr_ema = std::numeric_limits<float>::quiet_NaN();
    float classify_smoothed_chr_sum = 0.0f;
    int   classify_smoothed_chr_count = 0;
    float classify_chr_mean = std::numeric_limits<float>::quiet_NaN();
    const bool direct_classifier_threshold_enabled = std::isfinite(early_stop_ratio);
    const bool super_easy_policy_enabled =
        !paper_bucket_mode
        && direct_classifier_threshold_enabled
        && std::isfinite(super_easy_gamma_ratio);
    const bool mid_easy_bucket_policy_enabled =
        !paper_bucket_mode
        && direct_classifier_threshold_enabled
        && std::isfinite(mid_easy_upper_gamma_ratio);
    static constexpr bool ENABLE_ONE_SHOT_EFFECTIVE_EF_SHRINK = true;

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::vector<std::pair<dist_t, tableint>> top_container;
    top_container.reserve(ef_cur + 1);
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst> top_candidates(CompareByFirst(), std::move(top_container));

    std::vector<std::pair<dist_t, tableint>> candidate_container;
    candidate_container.reserve(ef_cur * 2);
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst> candidate_set(CompareByFirst(), std::move(candidate_container));

    dist_t lowerBound;

    if constexpr (bare_bone_search) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidate_set.emplace(-dist, ep_id);
    } else {
        if (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
        auto current_node_pair = candidate_set.top();
        dist_t candidate_dist = -current_node_pair.first;

        if (candidate_dist > lowerBound && top_candidates.size() == ef_cur)
            break;

        candidate_set.pop();
        tableint curr_id = current_node_pair.second;
        pop_count++;

        SearchStepInfo step{};
        step.node_id = curr_id;
        step.result_set_size = top_candidates.size();
        step.is_full_pop_after = false;
        step.full_pop_count_after = 0;
        step.runtime_accepted_rate = std::numeric_limits<float>::quiet_NaN();
        step.runtime_chr = std::numeric_limits<float>::quiet_NaN();
        step.runtime_smoothed_chr = std::numeric_limits<float>::quiet_NaN();
        step.runtime_classify_chr_mean = std::numeric_limits<float>::quiet_NaN();
        step.runtime_classification_evaluated = false;
        step.runtime_is_easy_query = false;
        step.runtime_is_super_easy_query = false;
        step.runtime_is_mid_easy_query = false;
        step.runtime_effective_ef = ef_cur;
        step.popped_query_dist = (float)candidate_dist;
        if (!top_candidates.empty()) {
            step.internal_dist = (float)top_candidates.top().first;
        }

        int *data = (int*)get_linklist0(curr_id);
        size_t size = getListCount((linklistsizeint*)data);
        size_t unvisited_count = 0;
        size_t accepted_count = 0;
        tableint *datal = (tableint *)(data + 1);
        step.popped_degree = size;

#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            tableint cand_id = *(datal + j);
#ifdef USE_SSE
            if (j + 1 < size) {
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
            }
#endif
            if (visited_array[cand_id] == visited_array_tag) continue;
            visited_array[cand_id] = visited_array_tag;
            unvisited_count++;

            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand_id), dist_func_param_);

            if (top_candidates.size() < ef_cur || lowerBound > d) {
                candidate_set.emplace(-d, cand_id);
                accepted_count++;
#ifdef USE_SSE
                _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);
#endif

                if constexpr (bare_bone_search) {
                    top_candidates.emplace(d, cand_id);
                } else {
                    if (!isMarkedDeleted(cand_id)) {
                        if (!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand_id))) {
                            top_candidates.emplace(d, cand_id);
                        }
                    }
                }

                if (top_candidates.size() > ef_cur)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }

        step.unvisited_neighbor_count = unvisited_count;
        step.accepted_neighbor_count = accepted_count;
        const float accepted_rate =
            (unvisited_count > 0)
                ? ((float)accepted_count / (float)unvisited_count)
                : 0.0f;

        if (top_candidates.size() == ef_cur) {
            step.is_full_pop_after = true;
            full_pop_count++;
            step.full_pop_count_after = (size_t)full_pop_count;
            step.runtime_accepted_rate = accepted_rate;
            float furthest_dist = (float)top_candidates.top().first;
            float chr = (float)candidate_dist / std::max(furthest_dist, 1e-6f);
            step.runtime_chr = chr;
            if (std::isnan(smoothed_chr_ema)) {
                smoothed_chr_ema = chr;
            } else {
                smoothed_chr_ema = CHR_EMA_DECAY * smoothed_chr_ema + CHR_EMA_UPDATE * chr;
            }
            step.runtime_smoothed_chr = smoothed_chr_ema;

            if (full_pop_count >= CLASSIFY_START && full_pop_count <= CLASSIFY_END) {
                classify_smoothed_chr_sum += smoothed_chr_ema;
                classify_smoothed_chr_count++;

                if (!classification_evaluated && full_pop_count == CLASSIFY_END) {
                    classification_evaluated = true;
                    if (direct_classifier_threshold_enabled) {
                        classify_chr_mean =
                            classify_smoothed_chr_sum / (float)classify_smoothed_chr_count;
                        is_easy_query = classify_chr_mean <= early_stop_ratio;
                        if (is_easy_query) {
                            float classify_chr_ratio =
                                classify_chr_mean / std::max(early_stop_ratio, 1e-6f);
                            if constexpr (!paper_bucket_mode) {
                                if (super_easy_policy_enabled) {
                                    is_super_easy_query = classify_chr_ratio <= super_easy_gamma_ratio;
                                }
                                if (mid_easy_bucket_policy_enabled) {
                                    is_mid_easy_query = classify_chr_ratio <= mid_easy_upper_gamma_ratio;
                                }
                            }
                        }
                    }

                    if (ENABLE_ONE_SHOT_EFFECTIVE_EF_SHRINK && !effective_ef_shrink_applied) {
                        size_t shrunk_ef_cur = configured_ef_cur;
                        if constexpr (paper_bucket_mode) {
                            if (is_easy_query) {
                                const float classify_chr_ratio =
                                    classify_chr_mean / std::max(early_stop_ratio, 1e-6f);
                                shrunk_ef_cur = resolvePaperBucketShrinkEf(
                                    configured_ef_cur,
                                    k,
                                    paper_bucket_count,
                                    classify_chr_ratio,
                                    paper_bucket_gamma_ratios
                                );
                            }
                        } else {
                            const size_t shrink_super_easy_ef =
                                resolveScaledShrinkEf(configured_ef_cur, 0.25, (size_t)128);
                            const size_t shrink_easy_ef =
                                std::max((size_t)1, configured_ef_cur / (size_t)2);
                            const size_t shrink_mid_easy_ef =
                                resolveScaledShrinkEf(configured_ef_cur, 0.50, (size_t)128);
                            const size_t shrink_edge_easy_ef =
                                resolveScaledShrinkEf(configured_ef_cur, 0.75, (size_t)256);

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
                            shrunk_ef_cur = std::max(shrunk_ef_cur, k);
                        }

                        if (shrunk_ef_cur < ef_cur) {
                            ef_cur = shrunk_ef_cur;
                            while (top_candidates.size() > ef_cur) {
                                top_candidates.pop();
                            }
                            if (!top_candidates.empty()) {
                                lowerBound = top_candidates.top().first;
                                furthest_dist = (float)top_candidates.top().first;
                            }
                        }
                        effective_ef_shrink_applied = true;
                    }
                }
            }

            step.runtime_classify_chr_mean = classify_chr_mean;
            step.runtime_classification_evaluated = classification_evaluated;
            step.runtime_is_easy_query = classification_evaluated && is_easy_query;
            step.runtime_is_super_easy_query = classification_evaluated && is_super_easy_query;
            step.runtime_is_mid_easy_query = classification_evaluated && is_mid_easy_query;
            step.runtime_effective_ef = ef_cur;
        }

        fillTraceStepMetrics(step, top_candidates, ef_cur, k, dim);
        output.path_info.push_back(std::move(step));

        if (stop_step > 0 && pop_count >= stop_step) {
            break;
        }
    }

    visited_list_pool_->releaseVisitedList(vl);

    while (top_candidates.size() > k)
        top_candidates.pop();

    std::vector<std::pair<dist_t, labeltype>> result_vec;
    result_vec.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
        result_vec.emplace_back(
            top_candidates.top().first,
            getExternalLabel(top_candidates.top().second)
        );
        top_candidates.pop();
    }

    output.stats.stop_count = 0;
    output.stats.reduced_steps = pop_count;
    output.result = std::priority_queue<std::pair<dist_t, labeltype>>(
        std::less<std::pair<dist_t, labeltype>>(),
        std::move(result_vec)
    );

    return output;
}



template <bool bare_bone_search>
TargetHitStepStats
searchBaseLayerBeamWidthFirstTargetHitStepCore(
    tableint ep_id,
    const void *data_point,
    size_t k,
    size_t ef_before,
    size_t switch_pop,
    size_t switch_full_pop,
    size_t ef_after,
    const labeltype* target_labels,
    size_t target_label_count,
    size_t target_hit_count,
    BaseFilterFunctor* isIdAllowed
) const {
    TargetHitStepStats stats;
    stats.target_hit_count = target_hit_count;
    if (target_hit_count == 0) {
        stats.reached_target = 1;
        return stats;
    }

    const size_t normalized_ef_before = std::max<size_t>(ef_before, k);
    const size_t normalized_ef_after = std::max<size_t>(ef_after, k);
    size_t ef_cur = normalized_ef_before;
    const size_t reserve_ef = std::max(normalized_ef_before, normalized_ef_after);
    bool phase_switch_applied = false;
    size_t pop_count = 0;
    size_t full_pop_count = 0;

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::vector<std::pair<dist_t, tableint>> top_container;
    top_container.reserve(reserve_ef + 1);
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst> top_candidates(CompareByFirst(), std::move(top_container));

    std::vector<std::pair<dist_t, tableint>> candidate_container;
    candidate_container.reserve(reserve_ef * 2);
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst> candidate_set(CompareByFirst(), std::move(candidate_container));

    dist_t lowerBound;

    if constexpr (bare_bone_search) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidate_set.emplace(-dist, ep_id);
    } else {
        if (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
        auto current_node_pair = candidate_set.top();
        dist_t candidate_dist = -current_node_pair.first;

        if (candidate_dist > lowerBound && top_candidates.size() == ef_cur)
            break;

        candidate_set.pop();
        tableint curr_id = current_node_pair.second;
        pop_count++;

        int *data = (int*)get_linklist0(curr_id);
        size_t size = getListCount((linklistsizeint*)data);
        tableint *datal = (tableint *)(data + 1);

#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            tableint cand_id = *(datal + j);
#ifdef USE_SSE
            if (j + 1 < size) {
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
            }
#endif
            if (visited_array[cand_id] == visited_array_tag) continue;
            visited_array[cand_id] = visited_array_tag;

            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand_id), dist_func_param_);

            if (top_candidates.size() < ef_cur || lowerBound > d) {
                candidate_set.emplace(-d, cand_id);
#ifdef USE_SSE
                _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);
#endif

                if constexpr (bare_bone_search) {
                    top_candidates.emplace(d, cand_id);
                } else {
                    if (!isMarkedDeleted(cand_id)) {
                        if (!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand_id))) {
                            top_candidates.emplace(d, cand_id);
                        }
                    }
                }

                if (top_candidates.size() > ef_cur)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }

        if (!phase_switch_applied && switch_pop > 0 && pop_count >= switch_pop) {
            phase_switch_applied = true;
            const size_t prev_ef = ef_cur;
            ef_cur = normalized_ef_after;
            if (ef_cur < prev_ef) {
                while (top_candidates.size() > ef_cur) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lowerBound = top_candidates.top().first;
                }
            }
        }

        if (top_candidates.size() == ef_cur) {
            full_pop_count++;
            if (!phase_switch_applied && switch_full_pop > 0 && full_pop_count >= switch_full_pop) {
                phase_switch_applied = true;
                const size_t prev_ef = ef_cur;
                ef_cur = normalized_ef_after;
                if (ef_cur < prev_ef) {
                    while (top_candidates.size() > ef_cur) {
                        top_candidates.pop();
                    }
                    if (!top_candidates.empty()) {
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        const size_t current_hit_count = computeTopKTargetHitCount(
            top_candidates,
            k,
            target_labels,
            target_label_count
        );
        stats.achieved_hit_count = std::max(stats.achieved_hit_count, current_hit_count);
        if (current_hit_count >= target_hit_count) {
            stats.first_target_hit_step = pop_count;
            stats.reached_target = 1;
            break;
        }
    }

    if (!stats.reached_target) {
        stats.achieved_hit_count = std::max(
            stats.achieved_hit_count,
            computeTopKTargetHitCount(top_candidates, k, target_labels, target_label_count)
        );
    }

    visited_list_pool_->releaseVisitedList(vl);
    return stats;
}

template <bool bare_bone_search, bool paper_bucket_mode>
std::priority_queue<std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    CompareByFirst>
searchBaseLayerAdaptiveLightCore(
    tableint ep_id,
    const void *data_point,
    size_t k,
    size_t ef_init,
    size_t ef_max,
    bool   enable_stop,
    size_t tmin_pops,
    float  early_stop_ratio,
    BaseFilterFunctor* isIdAllowed,
    float  super_easy_gamma_ratio,
    float  mid_easy_upper_gamma_ratio,
    size_t paper_bucket_count,
    const std::vector<float>& paper_bucket_gamma_ratios,
    int    classify_start = 4,
    int    classify_end = 16,
    float  chr_ema_decay = 0.8f
) const {
    size_t ef_cur = std::max<size_t>(ef_init, k);
    ef_cur = std::min(ef_cur, ef_max);
    const size_t configured_ef_cur = ef_cur;

    bool   is_easy_query    = false;
    bool   is_super_easy_query = false;
    bool   is_mid_easy_query = false;
    bool   classification_evaluated = false;
    bool   effective_ef_shrink_applied = false;
    int    full_pop_count   = 0;
    const int   CLASSIFY_START = classify_start;
    const int   CLASSIFY_END = classify_end;
    const float CHR_EMA_DECAY = chr_ema_decay;
    const float CHR_EMA_UPDATE = 1.0f - CHR_EMA_DECAY;

    float  smoothed_chr_ema = std::numeric_limits<float>::quiet_NaN();
    float  classify_smoothed_chr_sum = 0.0f;
    int    classify_smoothed_chr_count = 0;
    float  classify_chr_mean = std::numeric_limits<float>::quiet_NaN();
    const bool direct_classifier_threshold_enabled = std::isfinite(early_stop_ratio);
    const bool super_easy_policy_enabled =
        !paper_bucket_mode
        && direct_classifier_threshold_enabled
        && std::isfinite(super_easy_gamma_ratio);
    const bool mid_easy_bucket_policy_enabled =
        !paper_bucket_mode
        && direct_classifier_threshold_enabled
        && std::isfinite(mid_easy_upper_gamma_ratio);
    static constexpr bool   ENABLE_ONE_SHOT_EFFECTIVE_EF_SHRINK = true;

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::vector<std::pair<dist_t, tableint>> top_container;
    top_container.reserve(ef_cur + 1);
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst> top_candidates(CompareByFirst(), std::move(top_container));
    std::vector<std::pair<dist_t, tableint>> candidate_container;
    candidate_container.reserve(ef_cur * 2);
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst> candidate_set(CompareByFirst(), std::move(candidate_container));

    dist_t lowerBound;

    if constexpr (bare_bone_search) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidate_set.emplace(-dist, ep_id);
    } else {
        if (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
        auto current_node_pair = candidate_set.top();
        dist_t candidate_dist = -current_node_pair.first;

        if (candidate_dist > lowerBound && top_candidates.size() == ef_cur) {
            break;
        }

        candidate_set.pop();
        tableint curr_id = current_node_pair.second;

        int *data = (int*)get_linklist0(curr_id);
        size_t size = getListCount((linklistsizeint*)data);
        tableint *datal = (tableint *)(data + 1);

#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            tableint cand_id = *(datal + j);
#ifdef USE_SSE
            if (j + 1 < size) {
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
            }
#endif
            if (visited_array[cand_id] == visited_array_tag) continue;
            visited_array[cand_id] = visited_array_tag;

            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand_id), dist_func_param_);

            if (top_candidates.size() < ef_cur || lowerBound > d) {
                candidate_set.emplace(-d, cand_id);
#ifdef USE_SSE
                _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);
#endif

                if constexpr (bare_bone_search) {
                    top_candidates.emplace(d, cand_id);
                } else {
                    if (!isMarkedDeleted(cand_id)) {
                        if (!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand_id))) {
                            top_candidates.emplace(d, cand_id);
                        }
                    }
                }

                if (top_candidates.size() > ef_cur) {
                    top_candidates.pop();
                }

                if (!top_candidates.empty()) {
                    lowerBound = top_candidates.top().first;
                }
            }
        } // end for

        if (top_candidates.size() == ef_cur) {
            full_pop_count++;
            float furthest_dist = (float)top_candidates.top().first;
            float chr = (float)candidate_dist / std::max(furthest_dist, 1e-6f);
            if (std::isnan(smoothed_chr_ema)) {
                smoothed_chr_ema = chr;
            } else {
                smoothed_chr_ema = CHR_EMA_DECAY * smoothed_chr_ema + CHR_EMA_UPDATE * chr;
            }

            if (full_pop_count >= CLASSIFY_START && full_pop_count <= CLASSIFY_END) {
                classify_smoothed_chr_sum += smoothed_chr_ema;
                classify_smoothed_chr_count++;

                if (!classification_evaluated && full_pop_count == CLASSIFY_END) {
                    classification_evaluated = true;
                    if (direct_classifier_threshold_enabled) {
                        classify_chr_mean =
                            classify_smoothed_chr_sum / (float)classify_smoothed_chr_count;
                        is_easy_query = classify_chr_mean <= early_stop_ratio;
                        if (is_easy_query) {
                            float classify_chr_ratio =
                                classify_chr_mean / std::max(early_stop_ratio, 1e-6f);
                            if constexpr (!paper_bucket_mode) {
                                if (super_easy_policy_enabled) {
                                    is_super_easy_query = classify_chr_ratio <= super_easy_gamma_ratio;
                                }
                                if (mid_easy_bucket_policy_enabled) {
                                    is_mid_easy_query = classify_chr_ratio <= mid_easy_upper_gamma_ratio;
                                }
                            }
                        }
                    }

                    if (ENABLE_ONE_SHOT_EFFECTIVE_EF_SHRINK && !effective_ef_shrink_applied) {
                        size_t shrunk_ef_cur = configured_ef_cur;
                        if constexpr (paper_bucket_mode) {
                            if (is_easy_query) {
                                const float classify_chr_ratio =
                                    classify_chr_mean / std::max(early_stop_ratio, 1e-6f);
                                shrunk_ef_cur = resolvePaperBucketShrinkEf(
                                    configured_ef_cur,
                                    k,
                                    paper_bucket_count,
                                    classify_chr_ratio,
                                    paper_bucket_gamma_ratios
                                );
                            }
                        } else {
                            const size_t shrink_super_easy_ef =
                                resolveScaledShrinkEf(configured_ef_cur, 0.25, (size_t)128);
                            const size_t shrink_easy_ef =
                                std::max((size_t)1, configured_ef_cur / (size_t)2);
                            const size_t shrink_mid_easy_ef =
                                resolveScaledShrinkEf(configured_ef_cur, 0.50, (size_t)128);
                            const size_t shrink_edge_easy_ef =
                                resolveScaledShrinkEf(configured_ef_cur, 0.75, (size_t)256);
                            if (super_easy_policy_enabled && is_super_easy_query) {
                                shrunk_ef_cur = shrink_super_easy_ef;
                            } else if (is_easy_query) {
                                if (mid_easy_bucket_policy_enabled) {
                                    shrunk_ef_cur = is_mid_easy_query ? shrink_mid_easy_ef : shrink_edge_easy_ef;
                                } else {
                                    shrunk_ef_cur = shrink_easy_ef;
                                }
                            }
                            shrunk_ef_cur = std::max(shrunk_ef_cur, k);
                        }

                        if (shrunk_ef_cur < ef_cur) {
                            ef_cur = shrunk_ef_cur;
                            while (top_candidates.size() > ef_cur) {
                                top_candidates.pop();
                            }
                            if (!top_candidates.empty()) {
                                lowerBound = top_candidates.top().first;
                                furthest_dist = (float)top_candidates.top().first;
                            }
                        }
                        effective_ef_shrink_applied = true;
                    }
                }
            }

        }
    } // end while

    visited_list_pool_->releaseVisitedList(vl);

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }

    return top_candidates;
}




    template <bool bare_bone_search>
    AdaptiveSearchResult
    searchBaseLayerAdaptiveAnalysis(
        tableint ep_id,
        const void *data_point,
        size_t k,
        size_t ef_init,
        size_t ef_max,
        size_t tmin_pops,
        bool   enable_stop,
        size_t stop_step,
        BaseFilterFunctor* isIdAllowed = nullptr,
        float early_stop_ratio = 0.6f,
        float super_easy_gamma_ratio = std::numeric_limits<float>::quiet_NaN(),
        float mid_easy_upper_gamma_ratio = std::numeric_limits<float>::quiet_NaN(),
        int classify_start = 4,
        int classify_end = 16,
        float chr_ema_decay = 0.8f
    ) const {
        return searchBaseLayerAdaptiveAnalysisCore<bare_bone_search, false>(
            ep_id, data_point, k,
            ef_init, ef_max,
            tmin_pops,
            enable_stop,
            stop_step,
            early_stop_ratio,
            isIdAllowed,
            super_easy_gamma_ratio,
            mid_easy_upper_gamma_ratio,
            0,
            {},
            classify_start,
            classify_end,
            chr_ema_decay
        );
    }

    template <bool bare_bone_search>
    AdaptiveSearchResult
    searchBaseLayerAdaptiveAnalysisPaperBucket(
        tableint ep_id,
        const void *data_point,
        size_t k,
        size_t ef_init,
        size_t ef_max,
        size_t tmin_pops,
        bool   enable_stop,
        size_t stop_step,
        BaseFilterFunctor* isIdAllowed,
        float early_stop_ratio,
        size_t paper_bucket_count,
        const std::vector<float>& paper_bucket_gamma_ratios,
        int classify_start = 4,
        int classify_end = 16,
        float chr_ema_decay = 0.8f
    ) const {
        return searchBaseLayerAdaptiveAnalysisCore<bare_bone_search, true>(
            ep_id, data_point, k,
            ef_init, ef_max,
            tmin_pops,
            enable_stop,
            stop_step,
            early_stop_ratio,
            isIdAllowed,
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            paper_bucket_count,
            paper_bucket_gamma_ratios,
            classify_start,
            classify_end,
            chr_ema_decay
        );
    }

    template <bool bare_bone_search>
    std::priority_queue<std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    CompareByFirst>
    searchBaseLayerAdaptiveLight(
        tableint ep_id,
        const void *data_point,
        size_t k,
        size_t ef_init = 128,
        bool   enable_stop = true,
        BaseFilterFunctor* isIdAllowed = nullptr,
        float early_stop_ratio = 0.6f,
        size_t tmin_pops = 25,
        float super_easy_gamma_ratio = std::numeric_limits<float>::quiet_NaN(),
        float mid_easy_upper_gamma_ratio = std::numeric_limits<float>::quiet_NaN(),
        int classify_start = 4,
        int classify_end = 16,
        float chr_ema_decay = 0.8f
    ) const {
        constexpr size_t ef_max = 1024;
        return searchBaseLayerAdaptiveLightCore<bare_bone_search, false>(
            ep_id, data_point, k,
            ef_init, ef_max,
            enable_stop,
            tmin_pops,
            early_stop_ratio,
            isIdAllowed,
            super_easy_gamma_ratio,
            mid_easy_upper_gamma_ratio,
            0,
            {},
            classify_start,
            classify_end,
            chr_ema_decay
        );
    }

    template <bool bare_bone_search>
    std::priority_queue<std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    CompareByFirst>
    searchBaseLayerAdaptiveLightPaperBucket(
        tableint ep_id,
        const void *data_point,
        size_t k,
        size_t ef_init,
        bool   enable_stop,
        BaseFilterFunctor* isIdAllowed,
        float early_stop_ratio,
        size_t tmin_pops,
        size_t paper_bucket_count,
        const std::vector<float>& paper_bucket_gamma_ratios,
        int classify_start = 4,
        int classify_end = 16,
        float chr_ema_decay = 0.8f
    ) const {
        constexpr size_t ef_max = 1024;
        return searchBaseLayerAdaptiveLightCore<bare_bone_search, true>(
            ep_id, data_point, k,
            ef_init, ef_max,
            enable_stop,
            tmin_pops,
            early_stop_ratio,
            isIdAllowed,
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            paper_bucket_count,
            paper_bucket_gamma_ratios,
            classify_start,
            classify_end,
            chr_ema_decay
        );
    }

    TargetHitStepStats
    searchKnnBeamWidthFirstTargetHitStep(
        const void *query_data,
        size_t k,
        size_t ef_before,
        size_t switch_pop,
        size_t switch_full_pop,
        size_t ef_after,
        const labeltype* target_labels,
        size_t target_label_count,
        size_t target_hit_count,
        BaseFilterFunctor* isIdAllowed = nullptr
    ) const {
        TargetHitStepStats result;
        result.target_hit_count = target_hit_count;
        if (cur_element_count == 0) {
            return result;
        }

        tableint ep = getBaseLayerEntry(query_data);

        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            return searchBaseLayerBeamWidthFirstTargetHitStepCore<true>(
                ep,
                query_data,
                k,
                ef_before,
                switch_pop,
                switch_full_pop,
                ef_after,
                target_labels,
                target_label_count,
                target_hit_count,
                isIdAllowed
            );
        } else {
            return searchBaseLayerBeamWidthFirstTargetHitStepCore<false>(
                ep,
                query_data,
                k,
                ef_before,
                switch_pop,
                switch_full_pop,
                ef_after,
                target_labels,
                target_label_count,
                target_hit_count,
                isIdAllowed
            );
        }
    }

    AdaptiveSearchResult
    searchKnnAdaptiveAnalysis(
        const void *query_data,
        size_t k,
        size_t ef_init,
        size_t ef_max,
        size_t tmin_pops,
        bool   enable_stop,
        size_t stop_step = 0,
        BaseFilterFunctor* isIdAllowed = nullptr,
        float early_stop_ratio = 0.6f,
        float super_easy_gamma_ratio = std::numeric_limits<float>::quiet_NaN(),
        float mid_easy_upper_gamma_ratio = std::numeric_limits<float>::quiet_NaN(),
        int classify_start = 4,
        int classify_end = 16,
        float chr_ema_decay = 0.8f
    ) const {
        AdaptiveSearchResult result;
        if (cur_element_count == 0) {
            return result;
        }

        tableint ep = getBaseLayerEntry(query_data);

        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            return searchBaseLayerAdaptiveAnalysis<true>(
                ep, query_data, k,
                ef_init, ef_max,
                tmin_pops,
                enable_stop,
                stop_step,
                isIdAllowed,
                early_stop_ratio,
                super_easy_gamma_ratio,
                mid_easy_upper_gamma_ratio,
                classify_start,
                classify_end,
                chr_ema_decay
            );
        } else {
            return searchBaseLayerAdaptiveAnalysis<false>(
                ep, query_data, k,
                ef_init, ef_max,
                tmin_pops,
                enable_stop,
                stop_step,
                isIdAllowed,
                early_stop_ratio,
                super_easy_gamma_ratio,
                mid_easy_upper_gamma_ratio,
                classify_start,
                classify_end,
                chr_ema_decay
            );
        }
    }

    AdaptiveSearchResult
    searchKnnAdaptiveAnalysisPaperBucket(
        const void *query_data,
        size_t k,
        size_t ef_init,
        size_t ef_max,
        size_t tmin_pops,
        bool   enable_stop,
        size_t stop_step,
        BaseFilterFunctor* isIdAllowed,
        float early_stop_ratio,
        size_t paper_bucket_count,
        const std::vector<float>& paper_bucket_gamma_ratios,
        int classify_start = 4,
        int classify_end = 16,
        float chr_ema_decay = 0.8f
    ) const {
        AdaptiveSearchResult result;
        if (cur_element_count == 0) {
            return result;
        }

        tableint ep = getBaseLayerEntry(query_data);
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            return searchBaseLayerAdaptiveAnalysisPaperBucket<true>(
                ep, query_data, k,
                ef_init, ef_max,
                tmin_pops,
                enable_stop,
                stop_step,
                isIdAllowed,
                early_stop_ratio,
                paper_bucket_count,
                paper_bucket_gamma_ratios,
                classify_start,
                classify_end,
                chr_ema_decay
            );
        }
        return searchBaseLayerAdaptiveAnalysisPaperBucket<false>(
            ep, query_data, k,
            ef_init, ef_max,
            tmin_pops,
            enable_stop,
            stop_step,
            isIdAllowed,
            early_stop_ratio,
            paper_bucket_count,
            paper_bucket_gamma_ratios,
            classify_start,
            classify_end,
            chr_ema_decay
        );
    }

   std::priority_queue<std::pair<dist_t, labeltype>>
searchKnnAdaptiveLight(
    const void *query_data,
    size_t k,
    size_t ef_init = 128,
    bool   enable_stop = true,
    BaseFilterFunctor* isIdAllowed = nullptr,
    float early_stop_ratio = 0.6f,
    size_t tmin_pops = 25,
    float super_easy_gamma_ratio = std::numeric_limits<float>::quiet_NaN(),
    float mid_easy_upper_gamma_ratio = std::numeric_limits<float>::quiet_NaN(),
    int classify_start = 4,
    int classify_end = 16,
    float chr_ema_decay = 0.8f
) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) {
        return result;
    }

    tableint ep = getBaseLayerEntry(query_data);
    bool bare_bone_search = !num_deleted_ && !isIdAllowed;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

    if (bare_bone_search) {
        top_candidates = searchBaseLayerAdaptiveLight<true>(
            ep, query_data, k, ef_init, enable_stop, isIdAllowed, early_stop_ratio, tmin_pops,
            super_easy_gamma_ratio,
            mid_easy_upper_gamma_ratio,
            classify_start,
            classify_end,
            chr_ema_decay
        );
    } else {
        top_candidates = searchBaseLayerAdaptiveLight<false>(
            ep, query_data, k, ef_init, enable_stop, isIdAllowed, early_stop_ratio, tmin_pops,
            super_easy_gamma_ratio,
            mid_easy_upper_gamma_ratio,
            classify_start,
            classify_end,
            chr_ema_decay
        );
    }

    std::vector<std::pair<dist_t, labeltype>> result_vec;
    result_vec.reserve(top_candidates.size());

    while (!top_candidates.empty()) {
        result_vec.emplace_back(
            top_candidates.top().first,
            getExternalLabel(top_candidates.top().second)
        );
        top_candidates.pop();
    }

    return std::priority_queue<std::pair<dist_t, labeltype>>(
        std::less<std::pair<dist_t, labeltype>>(),
        std::move(result_vec)
    );
}

   std::priority_queue<std::pair<dist_t, labeltype>>
searchKnnAdaptiveLightPaperBucket(
    const void *query_data,
    size_t k,
    size_t ef_init,
    bool   enable_stop,
    BaseFilterFunctor* isIdAllowed,
    float early_stop_ratio,
    size_t tmin_pops,
    size_t paper_bucket_count,
    const std::vector<float>& paper_bucket_gamma_ratios,
    int classify_start = 4,
    int classify_end = 16,
    float chr_ema_decay = 0.8f
) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) {
        return result;
    }

    tableint ep = getBaseLayerEntry(query_data);
    bool bare_bone_search = !num_deleted_ && !isIdAllowed;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

    if (bare_bone_search) {
        top_candidates = searchBaseLayerAdaptiveLightPaperBucket<true>(
            ep,
            query_data,
            k,
            ef_init,
            enable_stop,
            isIdAllowed,
            early_stop_ratio,
            tmin_pops,
            paper_bucket_count,
            paper_bucket_gamma_ratios,
            classify_start,
            classify_end,
            chr_ema_decay
        );
    } else {
        top_candidates = searchBaseLayerAdaptiveLightPaperBucket<false>(
            ep,
            query_data,
            k,
            ef_init,
            enable_stop,
            isIdAllowed,
            early_stop_ratio,
            tmin_pops,
            paper_bucket_count,
            paper_bucket_gamma_ratios,
            classify_start,
            classify_end,
            chr_ema_decay
        );
    }

    std::vector<std::pair<dist_t, labeltype>> result_vec;
    result_vec.reserve(top_candidates.size());

    while (!top_candidates.empty()) {
        result_vec.emplace_back(
            top_candidates.top().first,
            getExternalLabel(top_candidates.top().second)
        );
        top_candidates.pop();
    }

    return std::priority_queue<std::pair<dist_t, labeltype>>(
        std::less<std::pair<dist_t, labeltype>>(),
        std::move(result_vec)
    );
}

    tableint getBaseLayerEntry(const void* query) const {
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query, getDataByInternalId(currObj), dist_func_param_);

        for (int level = maxlevel_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                auto* ll = get_linklist(currObj, level);
                size_t size = getListCount(ll);
                tableint* data = (tableint*)(ll + 1);

                for (size_t i = 0; i < size; i++) {
                    tableint cand = data[i];
                    dist_t d = fstdistfunc_(query, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        return currObj;
    }


    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
        tableint hidden_internal_id = HIDDEN_NODE_NONE) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (isHiddenNode(ep_id, hidden_internal_id)) {
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }
        if (bare_bone_search ||
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;
        if (hidden_internal_id != HIDDEN_NODE_NONE && hidden_internal_id < cur_element_count) {
            visited_array[hidden_internal_id] = visited_array_tag;
        }

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,
                                        _MM_HINT_T0);
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = (maxM0_ + 100) * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    *
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override {
        return searchKnnWithHiddenNode(query_data, k, isIdAllowed, HIDDEN_NODE_NONE);
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnWithHiddenNode(
        const void *query_data,
        size_t k,
        BaseFilterFunctor* isIdAllowed,
        tableint hidden_internal_id
    ) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = resolveEntryPointForHiddenNode(hidden_internal_id);
        if (isHiddenNode(currObj, hidden_internal_id)) {
            return result;
        }
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);

        int start_level = currObj == enterpoint_node_
            ? maxlevel_
            : std::min(maxlevel_, element_levels_[currObj]);
        for (int level = start_level; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (isHiddenNode(cand, hidden_internal_id)) {
                        continue;
                    }
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed && hidden_internal_id == HIDDEN_NODE_NONE;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed, nullptr, hidden_internal_id);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed, nullptr, hidden_internal_id);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};
}  // namespace hnswlib
