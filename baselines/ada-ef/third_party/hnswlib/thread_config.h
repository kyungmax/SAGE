#pragma once
#include <algorithm>
#include <thread>
namespace hnswdis {
// Zero keeps the upstream default; the standalone runner explicitly binds this.
inline unsigned configured_thread_count = 0;
inline void set_thread_count(unsigned count) { configured_thread_count = count; }
inline unsigned offline_thread_count() {
    return configured_thread_count ? configured_thread_count
        : std::max(1u, std::thread::hardware_concurrency() / 4);
}
}
