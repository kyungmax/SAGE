// Independent literal-set oracle for the optimized paper PiP bookkeeping.
#include <faiss/impl/PiPSetOverlap.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <vector>

using faiss::idx_t;

static double literal_overlap(
        const std::vector<idx_t>& before,
        const std::vector<idx_t>& after) {
    std::set<idx_t> a, b;
    for (auto id : before) {
        if (id >= 0) {
            a.insert(id);
        }
    }
    for (auto id : after) {
        if (id >= 0) {
            b.insert(id);
        }
    }
    size_t intersection = 0;
    for (auto id : a) {
        intersection += b.count(id);
    }
    return double(intersection) / before.size();
}

int main() {
    // Three accepted replacements can remove only one PREVIOUS ID. The two
    // transient IDs must not reduce the previous/current intersection again.
    std::vector<idx_t> labels{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    faiss::PiPSetOverlap overlap;
    overlap.initialize(labels.size(), labels.data());
    const auto before = labels;
    overlap.begin_step();
    for (idx_t fresh : {10, 11, 12}) {
        overlap.accepted_replacement(labels[0]);
        labels[0] = fresh;
    }
    assert(overlap.fraction() == 0.9);
    assert(overlap.fraction() == literal_overlap(before, labels));
    overlap.finish_step(labels.data());
    overlap.begin_step();
    assert(overlap.fraction() == 1.0);
    overlap.finish_step(labels.data());

    // Padding -1 is never a result ID, and the denominator remains k even
    // when fewer than k valid results exist at one or both checkpoints.
    labels = {0, -1, -1, -1};
    overlap.initialize(labels.size(), labels.data());
    overlap.begin_step();
    assert(overlap.fraction() == 0.25);
    overlap.accepted_replacement(-1);
    labels[1] = 1;
    assert(overlap.fraction() == 0.25);
    overlap.finish_step(labels.data());
    overlap.begin_step();
    assert(overlap.fraction() == 0.5);
    labels.assign(4, -1);
    overlap.initialize(labels.size(), labels.data());
    overlap.begin_step();
    assert(overlap.fraction() == 0.0);

    std::mt19937 rng(194731);
    size_t verified_steps = 0;
    for (size_t k : {size_t(1), size_t(2), size_t(10), size_t(100)}) {
        for (int trial = 0; trial < 200; ++trial) {
            idx_t next_id = 0;
            labels.assign(k, -1);
            const size_t initial_count = rng() % (k + 1);
            for (size_t i = 0; i < initial_count; ++i) {
                labels[i] = next_id++;
            }
            std::shuffle(labels.begin(), labels.end(), rng);
            overlap.initialize(k, labels.data());
            for (int step = 0; step < 50; ++step) {
                const auto previous = labels;
                overlap.begin_step();
                const int replacements = rng() % 9;
                for (int replacement = 0; replacement < replacements;
                     ++replacement) {
                    const size_t slot = rng() % k;
                    overlap.accepted_replacement(labels[slot]);
                    // Every admitted ID is new, matching the visited-table
                    // guarantee, including IDs transiently evicted this step.
                    labels[slot] = next_id++;
                }
                std::shuffle(labels.begin(), labels.end(), rng);
                assert(std::abs(overlap.fraction() -
                                literal_overlap(previous, labels)) < 1e-12);
                overlap.finish_step(labels.data());
                ++verified_steps;
            }
        }
    }
    std::cout << "{\"passed\":true,\"randomized_steps\":" << verified_steps
              << ",\"transient_replacement_case\":true,"
                 "\"underfilled_sentinel_case\":true}\n";
}
