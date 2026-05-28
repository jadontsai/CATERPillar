#include "gpu_candidate_selection.h"
#include "gpu_launch_config.h"


#include <cuda_runtime.h>

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

namespace{
__global__
void clear_selected_candidates_kernel(GpuSimulationState state) {
    int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (candidate_id >= state.candidates.total_candidates) {
        return;
    }
    //clears selected ID
    state.candidates.selected[candidate_id] = 0;
}
__global__ void clear_selected_by_front_kernel(
    GpuSimulationState state
) {
    int front_id = blockIdx.x * blockDim.x + threadIdx.x;
    int front_count = *state.fronts.count;

    if (front_id >= front_count) {
        return;
    }

    state.candidates.selected_by_front[front_id] = -1;
}
__global__
void select_valid_candidate_kernel(GpuSimulationState state){
    int front_id = blockIdx.x * blockDim.x + threadIdx.x;

    int front_count = *state.fronts.count;

    if (front_id >= front_count || state.fronts.active[front_id] == 0){
        return;
    }

        if (state.fronts.active[front_id] == 0) {
        return;
    }

    int candidates_per_front = state.candidates.candidates_per_front;
    int start = front_id * candidates_per_front;
    int end = start + candidates_per_front;

    for (int candidate_id = start; candidate_id < end; ++candidate_id) {
        if (state.candidates.valid[candidate_id] != 0) {
            //just make the first one valid, may change behaviour later
            //does introduce a bias towards the lower numbers
            state.candidates.selected[candidate_id] = 1;
            state.candidates.selected_by_front[front_id] = candidate_id;

            return;
        }
    }

}
}//end namespace

void select_valid_candidate_gpu(GpuSimulationState &state){
    int clear_blocks = gpu_num_blocks(state.candidates.total_candidates);

    clear_selected_candidates_kernel<<<clear_blocks, GPU_THREADS_PER_BLOCK>>>(state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int front_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &front_count,
        state.fronts.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (front_count <= 0) {
        return;
    }

    int select_blocks = gpu_num_blocks(front_count);
    clear_selected_by_front_kernel<<<select_blocks, GPU_THREADS_PER_BLOCK>>>(
        state
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    select_valid_candidate_kernel<<<select_blocks, GPU_THREADS_PER_BLOCK>>>(state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}