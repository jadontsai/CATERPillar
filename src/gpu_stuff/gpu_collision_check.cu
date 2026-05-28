#include "gpu_collision_check.h"
#include "gpu_launch_config.h"
#include "gpu_util.cuh"

#include <cuda_runtime.h>

#include <cmath>
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

namespace {
__device__
bool is_recent_ancestor(
    //since spheres can overlap with more than just 
    // their parent, check if it has an "ancestor" 
    // within some allowed range and allow those
    //  collisions (right now I'm using
    //  2 overlap factors, so 8, but variable radius 
    // might complicate that). doing this to avoid 
    // some weird looping stuff
    int sphere_idx,
    int parent_id,
    GpuSimulationState state,
    int skip_depth
) {
    int current = parent_id;

    for (int depth = 0; depth < skip_depth; ++depth) {
        if (current < 0) {
            return false;
        }
        // skip depth is probably 2*4, but that's a parameter that can be controlled
        if (sphere_idx == current) {
            return true;
        }

        current = state.spheres.parent_sphere_id[current];
    }

    return false;
}
__device__ bool overlap(float ax, float ay, float az, float ar,
float bx, float by, float bz, float br){
    float min_dist = ar+br;
    float dx = ax - bx;
    if (fabsf(dx) > min_dist) return false;

    float dy = ay - by;
    if (fabsf(dy) > min_dist)return false;

    float dz = az - bz;
    if (fabsf(dz) > min_dist) return false;
    float dist2 = dx * dx + dy * dy + dz * dz;
    return dist2 < min_dist * min_dist;
}
}
__global__
void spatial_grid_collision_check_kernel(
    GpuSimulationState state,
    GpuSpatialGrid grid
) {
    int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (candidate_idx >= state.candidates.total_candidates) {
        // out of bounds check
        return;
    }

    if (state.candidates.valid[candidate_idx] == 0) {
        // if invalid, skip
        return;
    }

    float cx = state.candidates.x[candidate_idx];
    float cy = state.candidates.y[candidate_idx];
    float cz = state.candidates.z[candidate_idx];
    float cr = state.candidates.r[candidate_idx];

    int parent_id = state.candidates.parent_id[candidate_idx];
    //skip depth, 8 for now as explained before
    //computing once per parent, slightly more efficient than before
    int skip_depth = static_cast<int>(ceilf(2.0f * state.params.overlap_factor));

    int x_min;
    int x_max;
    int y_min;
    int y_max;
    int z_min;
    int z_max;

    cell_bounds(
        cx,
        cy,
        cz,
        cr,
        grid,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max
    );

    for (int z = z_min; z <= z_max; ++z) {
        for (int y = y_min; y <= y_max; ++y) {
            for (int x = x_min; x <= x_max; ++x) {
                int flat_cell_idx = flatten_index(x, y, z, grid);
                if (flat_cell_idx < 0 || flat_cell_idx >= grid.num_cells) {
                    continue;
                }
                int count = grid.cell_counts[flat_cell_idx];
                count = min(count, grid.max_spheres_per_cell);
                for (int i =0; i<count;++i){
                    int sphere_idx = grid.cell_sphere_ids[flat_cell_idx
                    *grid.max_spheres_per_cell +i];
                    if (sphere_idx <0) continue;
                    if (is_recent_ancestor(
                        sphere_idx,
                        parent_id,
                        state,
                        skip_depth
                    )) {
                        continue;
                    }
                
                    float sx = state.spheres.x[sphere_idx];
                    float sy = state.spheres.y[sphere_idx];
                    float sz = state.spheres.z[sphere_idx];
                    float sr = state.spheres.r[sphere_idx];
                    //if touching logic:
                    if (overlap(cx,cy,cz,cr,sx,sy,sz,sr)){
                        state.candidates.valid[candidate_idx] = 0;
                        return;
                    }
                }
            }
        }
    } 
// __global__
// void spatial_grid_collision_check_kernel(
//     GpuSimulationState state,
//     GpuSpatialGrid grid
// ) {


// }

__global__ 
void selected_candidate_conflict_kernel(GpuSimulationState state){
    int front_id = blockIdx.x * blockDim.x + threadIdx.x;
    int front_count = *state.fronts.count;

    if (front_id >= front_count ||state.fronts.active[front_id] == 0) {
        return;
    }

    int cand_id = state.candidates.selected_by_front[front_id];
    if (state.candidates.selected[cand_id] == 0 ||
        state.candidates.valid[cand_id] == 0 ||cand_id < 0) {
        return;
    }

    float ix = state.candidates.x[cand_id];
    float iy = state.candidates.y[cand_id];
    float iz = state.candidates.z[cand_id];
    float ir = state.candidates.r[cand_id];

    for (int i = 0; i < front_id; ++i) {
        if (state.fronts.active[i] == 0) {
            continue;
        }
        int cand_2 = state.candidates.selected_by_front[i];

        if (cand_2 < 0) {
            continue;
        }

        if (state.candidates.selected[cand_2] == 0 ||
            state.candidates.valid[cand_2] == 0) {
            continue;
        }

        float jx = state.candidates.x[cand_2];
        float jy = state.candidates.y[cand_2];
        float jz = state.candidates.z[cand_2];
        float jr = state.candidates.r[cand_2];

        if (overlap(ix, iy, iz, ir, jx, jy, jz, jr)) {
            state.candidates.valid[cand_i] = 0;
            state.candidates.selected[cand_i] = 0;
            state.candidates.selected_by_front[front_i] = -1;
            return;
        }
    }
}
}// namespace

void run_collision_check(
    GpuSimulationState& state,
    GpuSpatialGrid& grid
) {
    int blocks = gpu_num_blocks(state.candidates.total_candidates);

    spatial_grid_collision_check_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(
        state,
        grid
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void run_selected_candidate_conflict_check(
    GpuSimulationState& state
) {
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

    int blocks = gpu_num_blocks(front_count);

    selected_candidate_conflict_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(
        state
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}