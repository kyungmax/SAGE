#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <cstring>
#include <vector>

#include <pybind11/stl.h>

#include "bindings_common.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

class HnswIndex {
   public:
    HnswIndex(
        size_t dim,
        size_t max_elements,
        size_t M,
        size_t ef_construction,
        size_t nbits,
        const std::string& metric = "l2",
        size_t random_seed = 100
    )
        : dim_(dim)
        , max_elements_(max_elements)
        , M_(M)
        , ef_construction_(ef_construction)
        , nbits_(nbits)
        , metric_(metric_from_string(metric))
        , random_seed_(random_seed)
        , index_(std::make_unique<rabitqlib::hnsw::HierarchicalNSW>(
              max_elements,
              dim,
              nbits,
              M,
              ef_construction,
              random_seed,
              metric_
          )) {}

    std::vector<rabitqlib::PID> resolve_hidden_internal_ids(py::object hide_labels, size_t nq) const {
        std::vector<rabitqlib::PID> hidden(nq, rabitqlib::kPidMax);
        if (hide_labels.is_none()) {
            return hidden;
        }
        auto labels_array = ensure_1d_array<rabitqlib::PID>(hide_labels, "hide_labels");
        if (static_cast<size_t>(labels_array.shape(0)) != nq) {
            throw std::invalid_argument("hide_labels length must match number of queries");
        }
        for (size_t i = 0; i < nq; ++i) {
            const rabitqlib::PID label = labels_array.data()[i];
            const rabitqlib::PID internal = index_->internal_id_for_label(label);
            if (internal == rabitqlib::kPidMax) {
                throw std::invalid_argument("hide_labels contains a label that is not present in the index");
            }
            hidden[i] = internal;
        }
        return hidden;
    }

    void build(
        py::handle data,
        py::handle centroids,
        py::handle cluster_ids,
        size_t num_threads = 1,
        bool fast_quantization = false
    ) {
        auto data_array = ensure_2d_array<float>(data, "data");
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");
        auto cluster_ids_array = ensure_1d_array<rabitqlib::PID>(cluster_ids, "cluster_ids");

        if (static_cast<size_t>(data_array.shape(1)) != dim_) {
            throw std::invalid_argument("data dimension does not match index dim");
        }
        if (static_cast<size_t>(centroids_array.shape(1)) != dim_) {
            throw std::invalid_argument("centroid dimension does not match index dim");
        }
        if (static_cast<size_t>(cluster_ids_array.shape(0)) != static_cast<size_t>(data_array.shape(0))) {
            throw std::invalid_argument("cluster_ids length must match number of rows in data");
        }

        const size_t num_clusters = static_cast<size_t>(centroids_array.shape(0));
        num_clusters_ = num_clusters;

        // Ensure cluster_ids are writable for the C++ API by making a copy
        std::vector<rabitqlib::PID> cluster_ids_vec(static_cast<size_t>(cluster_ids_array.shape(0)));
        std::memcpy(cluster_ids_vec.data(), cluster_ids_array.data(), cluster_ids_vec.size() * sizeof(rabitqlib::PID));

        index_->construct(
            num_clusters,
            centroids_array.data(),
            static_cast<size_t>(data_array.shape(0)),
            data_array.data(),
            cluster_ids_vec.data(),
            num_threads,
            fast_quantization
        );
        num_points_ = static_cast<size_t>(data_array.shape(0));
        built_ = true;
    }

    py::tuple search(py::handle queries, size_t k, size_t ef = 0, size_t num_threads = 1) {
        if (!built_ || k == 0 || num_threads == 0) {
            throw std::invalid_argument("Search requires a built index, positive k and num_threads");
        }
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (dim_ != 0 && static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (ef == 0) {
            ef = std::max<size_t>(k, 10);
        }

        const auto shape = std::vector<ssize_t>{
            static_cast<ssize_t>(query_array.shape(0)), static_cast<ssize_t>(k)};
        auto ids = py::array_t<rabitqlib::PID>(shape);
        auto dists = py::array_t<float>(shape);
        std::fill_n(ids.mutable_data(), ids.size(), rabitqlib::kPidMax);
        std::fill_n(dists.mutable_data(), dists.size(), std::numeric_limits<float>::infinity());
        auto ids_buf = ids.mutable_unchecked<2>();
        auto dists_buf = dists.mutable_unchecked<2>();

        std::vector<std::vector<std::pair<float, rabitqlib::PID>>> results = index_->search(
                query_array.data(),
                static_cast<size_t>(query_array.shape(0)),
                k,
                ef,
                num_threads
            );

        for (ssize_t i = 0; i < static_cast<ssize_t>(results.size()); ++i) {
            for (
                ssize_t j = 0;
                j < static_cast<ssize_t>(std::min<size_t>(k, results[static_cast<size_t>(i)].size()));
                ++j
            ) {
                ids_buf(i, j) = results[static_cast<size_t>(i)][static_cast<size_t>(j)].second;
                dists_buf(i, j) = results[static_cast<size_t>(i)][static_cast<size_t>(j)].first;
            }
        }
        return py::make_tuple(ids, dists);
    }

    py::tuple search_adaptive_light(
        py::handle queries,
        size_t k,
        size_t ef_init = 128,
        bool enable_stop = true,
        size_t num_threads = 1,
        float early_stop_ratio = 0.6F,
        bool paper_bucket_mode = false,
        int paper_bucket_count = 4,
        const std::vector<float>& bucket_gamma_ratios = {},
        py::object hide_labels = py::none(),
        int classify_start = 1,
        int classify_end = 16,
        bool shadow_stop_enabled = false,
        size_t shadow_ef = 128,
        float shadow_early_stop_ratio = 0.6F,
        bool shadow_bucket_mode = true,
        const std::vector<float>& shadow_bucket_gamma_ratios = {1.0F, 1.0F, 1.0F},
        bool shadow_pre_expansion = true
    ) {
        if (!built_ || k == 0 || num_threads == 0) {
            throw std::invalid_argument("Search requires a built index, positive k and num_threads");
        }
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (dim_ != 0 && static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (k == 0) {
            throw std::invalid_argument("k must be positive");
        }
        if (ef_init == 0) {
            ef_init = std::max<size_t>(k, 10);
        }
        if (bucket_gamma_ratios.size() > 7) {
            throw std::invalid_argument("bucket_gamma_ratios can contain at most 7 values");
        }

        rabitqlib::hnsw::AdaptiveLightConfig config;
        config.enable_stop = enable_stop;
        config.early_stop_ratio = early_stop_ratio;
        config.classify_start = classify_start;
        config.classify_end = classify_end;
        config.paper_bucket_mode = paper_bucket_mode;
        config.paper_bucket_count = paper_bucket_count;
        if (shadow_bucket_gamma_ratios.size() != 3) {
            throw std::invalid_argument("shadow_bucket_gamma_ratios requires three values");
        }
        config.shadow_stop_enabled = shadow_stop_enabled;
        config.shadow_pre_expansion = shadow_pre_expansion;
        config.shadow_ef = shadow_ef;
        config.shadow_early_stop_ratio = shadow_early_stop_ratio;
        config.shadow_bucket_mode = shadow_bucket_mode;
        std::copy(shadow_bucket_gamma_ratios.begin(), shadow_bucket_gamma_ratios.end(),
                  config.shadow_bucket_gamma_ratios.begin());
        for (size_t i = 0; i < bucket_gamma_ratios.size(); ++i) {
            config.bucket_gamma_ratios[i] = bucket_gamma_ratios[i];
        }

        const auto shape = std::vector<ssize_t>{
            static_cast<ssize_t>(query_array.shape(0)), static_cast<ssize_t>(k)};
        auto ids = py::array_t<rabitqlib::PID>(shape);
        auto dists = py::array_t<float>(shape);
        std::fill_n(ids.mutable_data(), ids.size(), rabitqlib::kPidMax);
        std::fill_n(dists.mutable_data(), dists.size(), std::numeric_limits<float>::infinity());
        auto ids_buf = ids.mutable_unchecked<2>();
        auto dists_buf = dists.mutable_unchecked<2>();

        const size_t nq = static_cast<size_t>(query_array.shape(0));
        std::vector<rabitqlib::PID> hidden_internal_ids =
            resolve_hidden_internal_ids(hide_labels, nq);
        std::vector<std::vector<std::pair<float, rabitqlib::PID>>> results =
            index_->search_adaptive_light(
                query_array.data(),
                nq,
                k,
                ef_init,
                num_threads,
                config,
                hide_labels.is_none() ? nullptr : hidden_internal_ids.data()
            );

        for (ssize_t i = 0; i < static_cast<ssize_t>(results.size()); ++i) {
            for (
                ssize_t j = 0;
                j < static_cast<ssize_t>(
                        std::min<size_t>(k, results[static_cast<size_t>(i)].size()));
                ++j
            ) {
                ids_buf(i, j) = results[static_cast<size_t>(i)][static_cast<size_t>(j)].second;
                dists_buf(i, j) = results[static_cast<size_t>(i)][static_cast<size_t>(j)].first;
            }
        }
        return py::make_tuple(ids, dists);
    }

    py::tuple search_adaptive_light_with_stats(
        py::handle queries,
        size_t k,
        size_t ef_init = 128,
        bool enable_stop = true,
        size_t num_threads = 1,
        float early_stop_ratio = 0.6F,
        bool paper_bucket_mode = false,
        int paper_bucket_count = 4,
        const std::vector<float>& bucket_gamma_ratios = {},
        py::object hide_labels = py::none(),
        int classify_start = 1,
        int classify_end = 16,
        bool shadow_stop_enabled = false,
        size_t shadow_ef = 128,
        float shadow_early_stop_ratio = 0.6F,
        bool shadow_bucket_mode = true,
        const std::vector<float>& shadow_bucket_gamma_ratios = {1.0F, 1.0F, 1.0F},
        bool shadow_pre_expansion = true,
        bool shadow_observation = false
    ) {
        if (!built_ || k == 0 || num_threads == 0) {
            throw std::invalid_argument("Search requires a built index, positive k and num_threads");
        }
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (dim_ != 0 && static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (k == 0) {
            throw std::invalid_argument("k must be positive");
        }
        if (ef_init == 0) {
            ef_init = std::max<size_t>(k, 10);
        }
        if (bucket_gamma_ratios.size() > 7) {
            throw std::invalid_argument("bucket_gamma_ratios can contain at most 7 values");
        }

        rabitqlib::hnsw::AdaptiveLightConfig config;
        config.enable_stop = enable_stop;
        config.early_stop_ratio = early_stop_ratio;
        config.classify_start = classify_start;
        config.classify_end = classify_end;
        config.paper_bucket_mode = paper_bucket_mode;
        config.paper_bucket_count = paper_bucket_count;
        if (shadow_bucket_gamma_ratios.size() != 3) {
            throw std::invalid_argument("shadow_bucket_gamma_ratios requires three values");
        }
        config.shadow_stop_enabled = shadow_stop_enabled;
        config.shadow_pre_expansion = shadow_pre_expansion;
        config.shadow_observation = shadow_observation;
        if (shadow_observation && (ef_init < 128 || shadow_ef != 128)) {
            throw std::invalid_argument("Shadow observation requires ef_init >= 128 and shadow_ef=128");
        }
        config.shadow_ef = shadow_ef;
        config.shadow_early_stop_ratio = shadow_early_stop_ratio;
        config.shadow_bucket_mode = shadow_bucket_mode;
        std::copy(shadow_bucket_gamma_ratios.begin(), shadow_bucket_gamma_ratios.end(),
                  config.shadow_bucket_gamma_ratios.begin());
        for (size_t i = 0; i < bucket_gamma_ratios.size(); ++i) {
            config.bucket_gamma_ratios[i] = bucket_gamma_ratios[i];
        }

        const size_t nq = static_cast<size_t>(query_array.shape(0));
        std::vector<rabitqlib::PID> hidden_internal_ids =
            resolve_hidden_internal_ids(hide_labels, nq);
        const auto shape = std::vector<ssize_t>{
            static_cast<ssize_t>(nq), static_cast<ssize_t>(k)};
        auto ids = py::array_t<rabitqlib::PID>(shape);
        auto dists = py::array_t<float>(shape);
        std::fill_n(ids.mutable_data(), ids.size(), rabitqlib::kPidMax);
        std::fill_n(dists.mutable_data(), dists.size(), std::numeric_limits<float>::infinity());
        auto ids_buf = ids.mutable_unchecked<2>();
        auto dists_buf = dists.mutable_unchecked<2>();

        auto output = index_->search_adaptive_light_with_stats(
            query_array.data(),
            nq,
            k,
            ef_init,
            num_threads,
            config,
            hide_labels.is_none() ? nullptr : hidden_internal_ids.data()
        );

        for (ssize_t i = 0; i < static_cast<ssize_t>(output.results.size()); ++i) {
            for (
                ssize_t j = 0;
                j < static_cast<ssize_t>(
                        std::min<size_t>(k, output.results[static_cast<size_t>(i)].size()));
                ++j
            ) {
                ids_buf(i, j) = output.results[static_cast<size_t>(i)][static_cast<size_t>(j)].second;
                dists_buf(i, j) = output.results[static_cast<size_t>(i)][static_cast<size_t>(j)].first;
            }
        }

        const auto stats_shape = std::vector<ssize_t>{static_cast<ssize_t>(nq)};
        const auto trace_shape = std::vector<ssize_t>{
            static_cast<ssize_t>(nq),
            static_cast<ssize_t>(rabitqlib::hnsw::AdaptiveLightStats::kClassifyTraceLen)
        };
        auto initial_ef = py::array_t<size_t>(stats_shape);
        auto effective_ef = py::array_t<size_t>(stats_shape);
        auto full_pop_count = py::array_t<size_t>(stats_shape);
        auto base_bin_est_count = py::array_t<size_t>(stats_shape);
        auto base_full_est_count = py::array_t<size_t>(stats_shape);
        auto entry_full_est_count = py::array_t<size_t>(stats_shape);
        auto upper_bin_est_count = py::array_t<size_t>(stats_shape);
        auto upper_link_scan_count = py::array_t<size_t>(stats_shape);
        auto base_pop_count = py::array_t<size_t>(stats_shape);
        auto base_neighbor_scan_count = py::array_t<size_t>(stats_shape);
        auto base_visited_count = py::array_t<size_t>(stats_shape);
        auto base_candidate_insert_count = py::array_t<size_t>(stats_shape);
        auto base_result_insert_count = py::array_t<size_t>(stats_shape);
        auto total_time_ns = py::array_t<uint64_t>(stats_shape);
        auto preprocess_time_ns = py::array_t<uint64_t>(stats_shape);
        auto upper_layer_time_ns = py::array_t<uint64_t>(stats_shape);
        auto base_layer_time_ns = py::array_t<uint64_t>(stats_shape);
        auto upper_bin_est_time_ns = py::array_t<uint64_t>(stats_shape);
        auto base_bin_est_time_ns = py::array_t<uint64_t>(stats_shape);
        auto base_full_est_time_ns = py::array_t<uint64_t>(stats_shape);
        auto classified = py::array_t<bool>(stats_shape);
        auto easy_query = py::array_t<bool>(stats_shape);
        auto super_easy_query = py::array_t<bool>(stats_shape);
        auto mid_easy_query = py::array_t<bool>(stats_shape);
        auto ef_shrunk = py::array_t<bool>(stats_shape);
        auto classify_cfr_mean = py::array_t<float>(stats_shape);
        auto classify_popped_dist = py::array_t<float>(trace_shape);
        auto classify_furthest_dist = py::array_t<float>(trace_shape);
        auto classify_cfr = py::array_t<float>(trace_shape);

        auto initial_ef_buf = initial_ef.mutable_unchecked<1>();
        auto effective_ef_buf = effective_ef.mutable_unchecked<1>();
        auto full_pop_count_buf = full_pop_count.mutable_unchecked<1>();
        auto base_bin_est_count_buf = base_bin_est_count.mutable_unchecked<1>();
        auto base_full_est_count_buf = base_full_est_count.mutable_unchecked<1>();
        auto entry_full_est_count_buf = entry_full_est_count.mutable_unchecked<1>();
        auto upper_bin_est_count_buf = upper_bin_est_count.mutable_unchecked<1>();
        auto upper_link_scan_count_buf = upper_link_scan_count.mutable_unchecked<1>();
        auto base_pop_count_buf = base_pop_count.mutable_unchecked<1>();
        auto base_neighbor_scan_count_buf = base_neighbor_scan_count.mutable_unchecked<1>();
        auto base_visited_count_buf = base_visited_count.mutable_unchecked<1>();
        auto base_candidate_insert_count_buf = base_candidate_insert_count.mutable_unchecked<1>();
        auto base_result_insert_count_buf = base_result_insert_count.mutable_unchecked<1>();
        auto total_time_ns_buf = total_time_ns.mutable_unchecked<1>();
        auto preprocess_time_ns_buf = preprocess_time_ns.mutable_unchecked<1>();
        auto upper_layer_time_ns_buf = upper_layer_time_ns.mutable_unchecked<1>();
        auto base_layer_time_ns_buf = base_layer_time_ns.mutable_unchecked<1>();
        auto upper_bin_est_time_ns_buf = upper_bin_est_time_ns.mutable_unchecked<1>();
        auto base_bin_est_time_ns_buf = base_bin_est_time_ns.mutable_unchecked<1>();
        auto base_full_est_time_ns_buf = base_full_est_time_ns.mutable_unchecked<1>();
        auto classified_buf = classified.mutable_unchecked<1>();
        auto easy_query_buf = easy_query.mutable_unchecked<1>();
        auto super_easy_query_buf = super_easy_query.mutable_unchecked<1>();
        auto mid_easy_query_buf = mid_easy_query.mutable_unchecked<1>();
        auto ef_shrunk_buf = ef_shrunk.mutable_unchecked<1>();
        auto classify_cfr_mean_buf = classify_cfr_mean.mutable_unchecked<1>();
        auto classify_popped_dist_buf = classify_popped_dist.mutable_unchecked<2>();
        auto classify_furthest_dist_buf = classify_furthest_dist.mutable_unchecked<2>();
        auto classify_cfr_buf = classify_cfr.mutable_unchecked<2>();

        for (size_t i = 0; i < nq; ++i) {
            const auto& stat = output.stats[i];
            const auto row = static_cast<ssize_t>(i);
            initial_ef_buf(row) = stat.initial_ef;
            effective_ef_buf(row) = stat.effective_ef;
            full_pop_count_buf(row) = stat.full_pop_count;
            base_bin_est_count_buf(row) = stat.base_bin_est_count;
            base_full_est_count_buf(row) = stat.base_full_est_count;
            entry_full_est_count_buf(row) = stat.entry_full_est_count;
            upper_bin_est_count_buf(row) = stat.upper_bin_est_count;
            upper_link_scan_count_buf(row) = stat.upper_link_scan_count;
            base_pop_count_buf(row) = stat.base_pop_count;
            base_neighbor_scan_count_buf(row) = stat.base_neighbor_scan_count;
            base_visited_count_buf(row) = stat.base_visited_count;
            base_candidate_insert_count_buf(row) = stat.base_candidate_insert_count;
            base_result_insert_count_buf(row) = stat.base_result_insert_count;
            total_time_ns_buf(row) = stat.total_time_ns;
            preprocess_time_ns_buf(row) = stat.preprocess_time_ns;
            upper_layer_time_ns_buf(row) = stat.upper_layer_time_ns;
            base_layer_time_ns_buf(row) = stat.base_layer_time_ns;
            upper_bin_est_time_ns_buf(row) = stat.upper_bin_est_time_ns;
            base_bin_est_time_ns_buf(row) = stat.base_bin_est_time_ns;
            base_full_est_time_ns_buf(row) = stat.base_full_est_time_ns;
            classified_buf(row) = stat.classified;
            easy_query_buf(row) = stat.easy_query;
            super_easy_query_buf(row) = stat.super_easy_query;
            mid_easy_query_buf(row) = stat.mid_easy_query;
            ef_shrunk_buf(row) = stat.ef_shrunk;
            classify_cfr_mean_buf(row) = stat.classify_cfr_mean;
            for (int trace_i = 0; trace_i < rabitqlib::hnsw::AdaptiveLightStats::kClassifyTraceLen; ++trace_i) {
                const auto col = static_cast<ssize_t>(trace_i);
                const auto trace_idx = static_cast<size_t>(trace_i);
                classify_popped_dist_buf(row, col) = stat.classify_popped_dist[trace_idx];
                classify_furthest_dist_buf(row, col) = stat.classify_furthest_dist[trace_idx];
                classify_cfr_buf(row, col) = stat.classify_cfr[trace_idx];
            }
        }

        py::dict stats;
        stats["initial_ef"] = initial_ef;
        stats["effective_ef"] = effective_ef;
        stats["full_pop_count"] = full_pop_count;
        stats["base_bin_est_count"] = base_bin_est_count;
        stats["base_full_est_count"] = base_full_est_count;
        stats["entry_full_est_count"] = entry_full_est_count;
        stats["upper_bin_est_count"] = upper_bin_est_count;
        stats["upper_link_scan_count"] = upper_link_scan_count;
        stats["base_pop_count"] = base_pop_count;
        stats["base_neighbor_scan_count"] = base_neighbor_scan_count;
        stats["base_visited_count"] = base_visited_count;
        stats["base_candidate_insert_count"] = base_candidate_insert_count;
        stats["base_result_insert_count"] = base_result_insert_count;
        stats["total_time_ns"] = total_time_ns;
        stats["preprocess_time_ns"] = preprocess_time_ns;
        stats["upper_layer_time_ns"] = upper_layer_time_ns;
        stats["base_layer_time_ns"] = base_layer_time_ns;
        stats["upper_bin_est_time_ns"] = upper_bin_est_time_ns;
        stats["base_bin_est_time_ns"] = base_bin_est_time_ns;
        stats["base_full_est_time_ns"] = base_full_est_time_ns;
        stats["classified"] = classified;
        stats["easy_query"] = easy_query;
        stats["super_easy_query"] = super_easy_query;
        stats["mid_easy_query"] = mid_easy_query;
        stats["ef_shrunk"] = ef_shrunk;
        stats["classify_cfr_mean"] = classify_cfr_mean;
        stats["classify_popped_dist"] = classify_popped_dist;
        stats["classify_furthest_dist"] = classify_furthest_dist;
        stats["classify_cfr"] = classify_cfr;
        auto main_decision_step = py::array_t<size_t>(stats_shape);
        for (size_t i = 0; i < nq; ++i) main_decision_step.mutable_data()[i] = output.stats[i].main_decision_step;
        stats["main_decision_step"] = main_decision_step;
        auto bucket_cut_step = py::array_t<size_t>(stats_shape);
        auto bucket_cut_ef = py::array_t<size_t>(stats_shape);
        auto top1_promotion_step = py::array_t<size_t>(stats_shape);
        auto top1_promotion_ef = py::array_t<size_t>(stats_shape);
        auto top1_at_cut = py::array_t<float>(stats_shape);
        auto top1_at_promotion = py::array_t<float>(stats_shape);
        for (size_t i = 0; i < nq; ++i) {
            bucket_cut_step.mutable_data()[i] = output.stats[i].bucket_cut_step;
            bucket_cut_ef.mutable_data()[i] = output.stats[i].bucket_cut_ef;
            top1_promotion_step.mutable_data()[i] = output.stats[i].top1_promotion_step;
            top1_promotion_ef.mutable_data()[i] = output.stats[i].top1_promotion_ef;
            top1_at_cut.mutable_data()[i] = output.stats[i].top1_at_cut;
            top1_at_promotion.mutable_data()[i] = output.stats[i].top1_at_promotion;
        }
        stats["bucket_cut_step"] = bucket_cut_step;
        stats["bucket_cut_ef"] = bucket_cut_ef;
        stats["top1_promotion_step"] = top1_promotion_step;
        stats["top1_promotion_ef"] = top1_promotion_ef;
        stats["top1_at_cut"] = top1_at_cut;
        stats["top1_at_promotion"] = top1_at_promotion;
        auto shadow_decision_step = py::array_t<size_t>(stats_shape);
        for (size_t i = 0; i < nq; ++i) shadow_decision_step.mutable_data()[i] = output.stats[i].shadow_decision_step;
        stats["shadow_decision_step"] = shadow_decision_step;
        auto shadow_updates = py::array_t<size_t>(stats_shape);
        for (size_t i = 0; i < nq; ++i) shadow_updates.mutable_data()[i] = output.stats[i].shadow_updates;
        stats["shadow_updates"] = shadow_updates;
        auto shadow_updates_at_decision = py::array_t<size_t>(stats_shape);
        for (size_t i = 0; i < nq; ++i) shadow_updates_at_decision.mutable_data()[i] = output.stats[i].shadow_updates_at_decision;
        stats["shadow_updates_at_decision"] = shadow_updates_at_decision;
        auto shadow_observations = py::array_t<size_t>(stats_shape);
        for (size_t i = 0; i < nq; ++i) shadow_observations.mutable_data()[i] = output.stats[i].shadow_observations;
        stats["shadow_observations"] = shadow_observations;
        auto shadow_proposed_ef = py::array_t<size_t>(stats_shape);
        for (size_t i = 0; i < nq; ++i) shadow_proposed_ef.mutable_data()[i] = output.stats[i].shadow_proposed_ef;
        stats["shadow_proposed_ef"] = shadow_proposed_ef;
        auto ef_before_shadow = py::array_t<size_t>(stats_shape);
        for (size_t i = 0; i < nq; ++i) ef_before_shadow.mutable_data()[i] = output.stats[i].ef_before_shadow;
        stats["ef_before_shadow"] = ef_before_shadow;
        auto shadow_routes = py::array_t<bool>(stats_shape);
        for (size_t i = 0; i < nq; ++i) shadow_routes.mutable_data()[i] = output.stats[i].shadow_routes;
        stats["shadow_routes"] = shadow_routes;
        auto shadow_cfr = py::array_t<float>(stats_shape);
        for (size_t i = 0; i < nq; ++i) shadow_cfr.mutable_data()[i] = output.stats[i].shadow_cfr;
        stats["shadow_cfr"] = shadow_cfr;
        auto shadow_popped_dist = py::array_t<float>(trace_shape);
        for (size_t i = 0; i < nq; ++i) {
            std::copy(output.stats[i].shadow_popped_dist.begin(), output.stats[i].shadow_popped_dist.end(),
                      shadow_popped_dist.mutable_data() + i * rabitqlib::hnsw::AdaptiveLightStats::kClassifyTraceLen);
        }
        stats["shadow_popped_dist"] = shadow_popped_dist;
        auto shadow_boundary = py::array_t<float>(trace_shape);
        for (size_t i = 0; i < nq; ++i) {
            std::copy(output.stats[i].shadow_boundary.begin(), output.stats[i].shadow_boundary.end(),
                      shadow_boundary.mutable_data() + i * rabitqlib::hnsw::AdaptiveLightStats::kClassifyTraceLen);
        }
        stats["shadow_boundary"] = shadow_boundary;
        stats["final_ef"] = effective_ef;
        stats["hops"] = base_pop_count;
        stats["main_cfr"] = classify_cfr_mean;
        return py::make_tuple(ids, dists, stats);
    }

    py::array_t<float> pairwise_est_dist(py::handle queries) {
        if (!built_) {
            throw std::runtime_error(
                "HnswIndex must be built before pairwise_est_dist"
            );
        }

        auto query_array =
            ensure_2d_array<float>(queries, "queries");

        const size_t nq =
            static_cast<size_t>(query_array.shape(0));

        std::vector<float> flat =
            index_->pairwise_est_dist(
                query_array.data(),
                nq
            );

        const auto shape =
            std::vector<ssize_t>{
                static_cast<ssize_t>(nq),
                static_cast<ssize_t>(num_points_)
            };

        auto out = py::array_t<float>(shape);
        auto out_buf = out.mutable_unchecked<2>();

        for (size_t i = 0; i < nq; ++i) {
            for (size_t j = 0; j < num_points_; ++j) {
                out_buf(
                    static_cast<ssize_t>(i),
                    static_cast<ssize_t>(j)
                ) = flat[i * num_points_ + j];
            }
        }

        return out;
    }

    void save(const std::string& path) const {
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before save");
        }
        index_->save(path.c_str());
    }

    static HnswIndex load(const std::string& path) {
        HnswIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::hnsw::HierarchicalNSW>();
        wrapper.index_->load(path.c_str());
        wrapper.dim_ = wrapper.index_->dimension();
        wrapper.max_elements_ = wrapper.index_->max_elements();
        wrapper.M_ = wrapper.index_->M();
        wrapper.ef_construction_ = wrapper.index_->ef_construction();
        wrapper.nbits_ = wrapper.index_->nbits();
        wrapper.num_clusters_ = wrapper.index_->num_clusters();
        wrapper.num_points_ = wrapper.index_->current_count();
        wrapper.metric_ = wrapper.index_->metric_type();
        wrapper.built_ = true;
        return wrapper;
    }

    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t M() const { return M_; }
    [[nodiscard]] size_t ef_construction() const { return ef_construction_; }
    [[nodiscard]] size_t max_elements() const { return max_elements_; }
    [[nodiscard]] size_t num_points() const { return num_points_; }
    [[nodiscard]] size_t nbits() const { return nbits_; }
    [[nodiscard]] bool is_built() const { return built_; }
    [[nodiscard]] size_t num_clusters() const { return num_clusters_; }
    [[nodiscard]] std::string metric() const { return metric_to_string(metric_); }

   private:
    HnswIndex() = default;

    size_t dim_ = 0;
    size_t max_elements_ = 0;
    size_t M_ = 0;
    size_t ef_construction_ = 0;
    size_t nbits_ = 0;
    rabitqlib::MetricType metric_ = rabitqlib::METRIC_L2;
    size_t random_seed_ = 100;
    size_t num_clusters_ = 0;
    bool built_ = false;
    size_t num_points_ = 0;
    std::unique_ptr<rabitqlib::hnsw::HierarchicalNSW> index_;
};

}  // namespace rabitqlib::python_bindings

// Register into combined module
void register_hnsw(py::module_ &m) {
    using namespace rabitqlib::python_bindings;

    py::class_<HnswIndex>(m, "HnswIndex")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, const std::string&, size_t>(),
             py::arg("dim"),
             py::arg("max_elements"),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("nbits") = 8,
             py::arg("metric") = "l2",
             py::arg("random_seed") = 100)
        .def("build", &HnswIndex::build,
             py::arg("data"),
             py::arg("centroids"),
             py::arg("cluster_ids"),
             py::arg("num_threads") = 1,
             py::arg("fast_quantization") = false)
        .def("search", &HnswIndex::search,
             py::arg("queries"),
             py::arg("k"),
             py::arg("ef") = 0,
             py::arg("num_threads") = 1)
        .def("search_adaptive_light", &HnswIndex::search_adaptive_light,
             py::arg("queries"),
             py::arg("k"),
             py::arg("ef_init") = 128,
             py::arg("enable_stop") = true,
             py::arg("num_threads") = 1,
             py::arg("early_stop_ratio") = 0.6F,
             py::arg("paper_bucket_mode") = false,
             py::arg("paper_bucket_count") = 4,
             py::arg("bucket_gamma_ratios") = std::vector<float>{},
             py::arg("hide_labels") = py::none(),
             py::arg("classify_start") = 1,
             py::arg("classify_end") = 16,
             py::arg("shadow_stop_enabled") = false,
             py::arg("shadow_ef") = 128,
             py::arg("shadow_early_stop_ratio") = 0.6F,
             py::arg("shadow_bucket_mode") = true,
             py::arg("shadow_bucket_gamma_ratios") = std::vector<float>{1.0F, 1.0F, 1.0F},
             py::arg("shadow_pre_expansion") = true)
        .def("search_sage", &HnswIndex::search_adaptive_light,
             py::arg("queries"),
             py::arg("k"),
             py::arg("ef_init") = 128,
             py::arg("enable_stop") = true,
             py::arg("num_threads") = 1,
             py::arg("early_stop_ratio") = 0.6F,
             py::arg("paper_bucket_mode") = false,
             py::arg("paper_bucket_count") = 4,
             py::arg("bucket_gamma_ratios") = std::vector<float>{},
             py::arg("hide_labels") = py::none(),
             py::arg("classify_start") = 1,
             py::arg("classify_end") = 16,
             py::arg("shadow_stop_enabled") = false,
             py::arg("shadow_ef") = 128,
             py::arg("shadow_early_stop_ratio") = 0.6F,
             py::arg("shadow_bucket_mode") = true,
             py::arg("shadow_bucket_gamma_ratios") = std::vector<float>{1.0F, 1.0F, 1.0F},
             py::arg("shadow_pre_expansion") = true)
        .def("search_adaptive_light_with_stats", &HnswIndex::search_adaptive_light_with_stats,
             py::arg("queries"),
             py::arg("k"),
             py::arg("ef_init") = 128,
             py::arg("enable_stop") = true,
             py::arg("num_threads") = 1,
             py::arg("early_stop_ratio") = 0.6F,
             py::arg("paper_bucket_mode") = false,
             py::arg("paper_bucket_count") = 4,
             py::arg("bucket_gamma_ratios") = std::vector<float>{},
             py::arg("hide_labels") = py::none(),
             py::arg("classify_start") = 1,
             py::arg("classify_end") = 16,
             py::arg("shadow_stop_enabled") = false,
             py::arg("shadow_ef") = 128,
             py::arg("shadow_early_stop_ratio") = 0.6F,
             py::arg("shadow_bucket_mode") = true,
             py::arg("shadow_bucket_gamma_ratios") = std::vector<float>{1.0F, 1.0F, 1.0F},
             py::arg("shadow_pre_expansion") = true,
             py::arg("shadow_observation") = false)
        .def("pairwise_est_dist",
             &HnswIndex::pairwise_est_dist,
             py::arg("queries"))
        .def("save", &HnswIndex::save, py::arg("path"))
        .def_static("load", &HnswIndex::load, py::arg("path"))
        .def_property_readonly("dim", &HnswIndex::dim)
        .def_property_readonly("M", &HnswIndex::M)
        .def_property_readonly("ef_construction", &HnswIndex::ef_construction)
        .def_property_readonly("max_elements", &HnswIndex::max_elements)
        .def_property_readonly("num_points", &HnswIndex::num_points)
        .def_property_readonly("nbits", &HnswIndex::nbits)
        .def_property_readonly("num_clusters", &HnswIndex::num_clusters)
        .def_property_readonly("is_built", &HnswIndex::is_built)
        .def_property_readonly("metric", &HnswIndex::metric);
}
