#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <faiss/Index.h>

namespace faiss {

/** Exact previous/current top-k set intersection for unique visited IDs.
 * An accepted result evicts the heap root. Only evictions of IDs present at the
 * preceding checkpoint reduce the intersection. Transient new results that are
 * inserted and evicted within one expansion do not reduce it a second time.
 * The snapshot is refreshed only when the result heap changed; no per-step
 * allocation, heap mutation, or cumulative-insertion approximation is used.
 */
struct PiPSetOverlap {
    size_t k = 0;
    size_t retained = 0;
    bool changed = false;
    std::vector<idx_t> previous;

    void snapshot(const idx_t* labels) {
        previous.clear();
        for (size_t i = 0; i < k; ++i) {
            if (labels[i] >= 0) {
                previous.push_back(labels[i]);
            }
        }
    }

    void initialize(size_t count, const idx_t* labels) {
        k = count;
        previous.reserve(k);
        snapshot(labels);
    }

    void begin_step() {
        retained = previous.size();
        changed = false;
    }

    void accepted_replacement(idx_t evicted) {
        changed = true;
        if (evicted >= 0 &&
            std::find(previous.begin(), previous.end(), evicted) !=
                    previous.end()) {
            --retained;
        }
    }

    double fraction() const {
        return double(retained) / double(k);
    }

    void finish_step(const idx_t* labels) {
        if (changed) {
            snapshot(labels);
        }
    }
};

} // namespace faiss
