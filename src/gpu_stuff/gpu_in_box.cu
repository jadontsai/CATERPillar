#include "gpu_in_box.h"
#include "gpu_launch_config.h"
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
void in_box_check_kernel(GpuSimulationState state) {
    // Implementation for checking in box on GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state.candidates.total_candidates) {
        return;// out of bounds
    }

    if (state.candidates.valid[idx] == 0) {
        return;
    }
    //for less memory accesses, load into registers
    float x = state.candidates.x[idx];
    float y = state.candidates.y[idx];
    float z = state.candidates.z[idx];
    float r = state.candidates.r[idx];
    float voxel_edge = state.params.voxel_edge_length;

    bool in_box = (x-r >= 0.0f) && (x+r <= voxel_edge) &&
                  (y-r >= 0.0f) && (y+r <= voxel_edge) &&
                  (z-r >= 0.0f) && (z+r <= voxel_edge);

    if (!in_box) {
        state.candidates.valid[idx] = 0;
    }

}

void run_in_box_check(GpuSimulationState& state){
    int blocks = gpu_num_blocks(state.candidates.total_candidates);
    
    in_box_check_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    }
