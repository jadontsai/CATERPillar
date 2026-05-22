#pragma once

#include <stdexcept>

constexpr int GPU_THREADS_PER_BLOCK = 256;
constexpr int GPU_MAX_THREADS_PER_BLOCK = 1024;

inline int gpu_num_blocks(int num_threads) {

    if (num_threads <= 0) {
        return 0;
    }
    return (num_threads + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
}

inline void validate_threads_per_block(int threads_per_block) {
    if (threads_per_block <= 0 || threads_per_block > GPU_MAX_THREADS_PER_BLOCK) {
        throw std::invalid_argument("Threads per block must be between 1 and " + std::to_string(GPU_MAX_THREADS_PER_BLOCK));
    }
}