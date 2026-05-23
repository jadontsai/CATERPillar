#include <gpu_commit.h>
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
void commit_candidates_kernel(GpuSimulationState state) {
    int front_id = blockIdx.x * blockDim.x + threadIdx.x;
    int front_count = *state.fronts.count;
    if (front_id >= front_count || state.fronts.active[front_id] == 0) {
        //if the front does not exist or is inactive, skip it
        return;
    }
    int candidates_per_front = state.candidates.candidates_per_front;
    int start = front_id * candidates_per_front;
    int end = start + candidates_per_front;
      
    int selected_candidate_id = -1;

    for (int candidate_id = start; candidate_id < end; ++candidate_id) {
        if (state.candidates.valid[candidate_id] != 0) {
            selected_candidate_id = candidate_id;
            break;
        }
    }
    
    if (selected_candidate_id == -1) {
        //no valid candidate found for this front, skip it
        return;
    }

    int new_sphere_index = atomicAdd(state.spheres.count, 1);
    if (new_sphere_index >= state.params.max_spheres) {
        *state.error_code = 1; //overflow
        return;
    }

    int parent_sphere_id = state.candidates.parent_id[selected_candidate_id];
    state.spheres.x[new_sphere_index] = state.candidates.x[selected_candidate_id];
    state.spheres.y[new_sphere_index] = state.candidates.y[selected_candidate_id];
    state.spheres.z[new_sphere_index] = state.candidates.z[selected_candidate_id];
    state.spheres.r[new_sphere_index] = state.candidates.r[selected_candidate_id];
    state.spheres.object_type[new_sphere_index] = state.fronts.object_type[front_id]; //set as needed
    state.spheres.object_id[new_sphere_index] = state.fronts.object_id[front_id];  
    state.spheres.branch_id[new_sphere_index] = state.fronts.branch_id[front_id];
    state.spheres.parent_sphere_id[new_sphere_index] = parent_sphere_id;
  
    state.fronts.x[front_id] = state.candidates.x[selected_candidate_id];
    state.fronts.y[front_id] = state.candidates.y[selected_candidate_id];
    state.fronts.z[front_id] = state.candidates.z[selected_candidate_id];
    state.fronts.r[front_id] = state.candidates.r[selected_candidate_id];
    state.fronts.dir_x[front_id] = state.candidates.dir_x[selected_candidate_id];
    state.fronts.dir_y[front_id] = state.candidates.dir_y[selected_candidate_id];
    state.fronts.dir_z[front_id] = state.candidates.dir_z[selected_candidate_id];
    state.fronts.parent_sphere_id[front_id] = parent_sphere_id;
     }
}//end namespace


void commit_candidates_gpu(GpuSimulationState& state) {
    int front_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &front_count,
        state.fronts.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    if (front_count <= 0) {
        //something went wrong here
        return;
    }
    int blocks = gpu_num_blocks(front_count);

    commit_candidates_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
