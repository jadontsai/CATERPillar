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
//copy pasted a lot of the helper functions, they both live in namespaces and 
// i don't really want to undo this, should have way more than 
// enough memory with DRAC anyways
// (if that somehow becomes an issue i'll make a helper file)
//okay nvm making the helper file

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

                int start = grid.cell_start[flat_cell_idx];
                int end = grid.cell_end[flat_cell_idx];

                if (start < 0 || end < 0) {
                    continue;
                }

                for (int entry_idx = start; entry_idx < end; ++entry_idx) {
                    int sphere_idx = grid.grid_sphere_id[entry_idx];
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
                    float min_dist = cr + sr;

                    float dx = cx - sx;
                    if (fabsf(dx) > min_dist) {
                        continue;
                    }

                    float dy = cy - sy;
                    if (fabsf(dy) > min_dist) {
                        continue;
                    }

                    float dz = cz - sz;
                    if (fabsf(dz) > min_dist) {
                        continue;
                    }

                    float dist2 = dx * dx + dy * dy + dz * dz;
                    float min_dist2 = min_dist * min_dist;

                    if (dist2 < min_dist2) {
                        state.candidates.valid[candidate_idx] = 0;
                        return;
                    }
                }
            }
        }
    }
}

} // namespace

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