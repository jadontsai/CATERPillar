#include "gpu_commit.h"//oops
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
    int selected_candidate_id = state.candidates.selected_by_front[front_id];

    if (selected_candidate_id < 0) {
        return;
    }

    if (state.candidates.selected[selected_candidate_id] == 0 ||
        state.candidates.valid[selected_candidate_id] == 0) {
        return;
    }

    int new_sphere_index = atomicAdd(state.spheres.count, 1);

    if (new_sphere_index >= state.params.max_spheres) {
        atomicSub(state.spheres.count, 1);
        *state.error_code = 1; // sphere table overflow
        return;
    }

    int parent_sphere_id = state.candidates.parent_id[selected_candidate_id];
    float base_radius = state.fronts.base_r[front_id];
    state.spheres.x[new_sphere_index] = state.candidates.x[selected_candidate_id];
    state.spheres.y[new_sphere_index] = state.candidates.y[selected_candidate_id];
    state.spheres.z[new_sphere_index] = state.candidates.z[selected_candidate_id];
    state.spheres.r[new_sphere_index] = state.candidates.r[selected_candidate_id];
    state.spheres.base_r[new_sphere_index] = base_radius;
    state.spheres.object_type[new_sphere_index] = state.fronts.object_type[front_id]; //set as needed
    state.spheres.object_id[new_sphere_index] = state.fronts.object_id[front_id];  
    state.spheres.branch_id[new_sphere_index] = state.fronts.branch_id[front_id];
    state.spheres.parent_sphere_id[new_sphere_index] = parent_sphere_id;
  
    state.fronts.x[front_id] = state.candidates.x[selected_candidate_id];
    state.fronts.y[front_id] = state.candidates.y[selected_candidate_id];
    state.fronts.z[front_id] = state.candidates.z[selected_candidate_id];
    state.fronts.r[front_id] = state.candidates.r[selected_candidate_id];
    state.fronts.base_r[front_id] = base_radius;
    state.fronts.dir_x[front_id] = state.candidates.dir_x[selected_candidate_id];
    state.fronts.dir_y[front_id] = state.candidates.dir_y[selected_candidate_id];
    state.fronts.dir_z[front_id] = state.candidates.dir_z[selected_candidate_id];
    state.fronts.parent_sphere_id[front_id] = new_sphere_index;//made change here, needs to grow from new sphere
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

void commit_candidates_and_update_grid_gpu(
    GpuSimulationState& state,
    GpuSpatialGrid& grid
) {
    int old_sphere_count = 0;
    int new_sphere_count = 0;

    CUDA_CHECK(cudaMemcpy(
        &old_sphere_count,
        state.spheres.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    commit_candidates_gpu(state);

    CUDA_CHECK(cudaMemcpy(
        &new_sphere_count,
        state.spheres.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    insert_spheres_into_gpu_spatial_grid(
        state,
        grid,
        old_sphere_count,
        new_sphere_count
    );
}