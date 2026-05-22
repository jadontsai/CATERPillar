#include "gpu_in_box.h"

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
void in_box_check_kernel(const GpuSimulationState& state, bool* in_box_result) {
    // Implementation for checking in box on GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state.candidates.total_candidates) {
        return;// out of bounds
    }

    if (state.candidates.valid[idx] == 0) {
        in_box_result[idx] = false;// if it's not valid somehow already, no need to check
        return;
    }

    //for less memory accesses, load into registers
    float x = state.candidates.x[idx];
    float y = state.candidates.y[idx];
    float z = state.candidates.z[idx];
    float r = state.candidates.r[idx];
    float voxel_edge = state.params.voxel_edge;

    bool in_box = (x-r >= 0.0f) && (x+r <= voxel_edge) &&
                  (y-r >= 0.0f) && (y+r <= voxel_edge) &&
                  (z-r >= 0.0f) && (z+r <= voxel_edge);

    if (!in_box) {
        state.candidates.valid[idx] = false;
    }

}

void run_in_box_check(GpuSimulationState& state){
    int threads_per_block = 256;
    int blocks = (state.candidates.total_candidates + 
        threads_per_block - 1) / threads_per_block;
    
    in_box_check_kernel<<<blocks, threads_per_block>>>(state, state.candidates.valid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    }
