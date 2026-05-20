#include "gpu_collision_check.h"

#include <cuda_runtime.h>//for the usual cuda functions

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err));    \
        }                                                                  \
    } while (0)

__global__
void collision_check_kernel(const GpuSimulationState& sim_state, bool* collision_result) {
    // Implementation for collision checking on GPU

    
}
