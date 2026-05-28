#include "gpu_spatial_grid.h"
#include "gpu_launch_config.h"
#include "gpu_util.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>//for thrust::sort_by_key
#include <thrust/scan.h>//for thrust::exclusive_scan for prexix sum
#include <stdexcept>
#include <string>
#include <cmath>


//idea behind spatial grid is to divide voxel
//  into a grid of cells, and keep track of
//  which spheres are in which cells.
//this kernel just sorts the entries, 
//then another kernel (gpu_collision_check.cu) can do 
// collision checks by only checking spheres in close enough cells
// this is then used by gpu_collision_checker to actually check
//this keeps the complexity of collision checking within O(n) since it's akin
// to dictionary accesses
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
void num_sphere_grid_entries_kernel(//helper function to count number of grids a sphere touches
    GpuSimulationState state,
    GpuSpatialGrid grid,
    int begin,
    int end
)   
{   int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sphere_idx = local_idx + begin;

    //avoid out of bounds access, so excess threads exit early
    if (sphere_idx >= end) return;

    int x_min, x_max, y_min, y_max, z_min, z_max;
    cell_bounds(
        state.spheres.x[sphere_idx],
        state.spheres.y[sphere_idx],
        state.spheres.z[sphere_idx],
        state.spheres.r[sphere_idx],
        grid,
        x_min, x_max, y_min, y_max, z_min, z_max
    );
    for (int z = z_min; z <= z_max; ++z) {
        for (int y = y_min; y <= y_max; ++y) {
            for (int x = x_min; x <= x_max; ++x) {
                int cell_idx = flatten_index(x, y, z, grid);//to make things one dimensional

                if (cell_idx < 0 || cell_idx >= grid.num_cells) {
                    continue;
                }

                int slot = atomicAdd(&grid.cell_counts[cell_idx], 1);

                if (slot >= grid.max_spheres_per_cell) {
                    *state.error_code = 2; // spatial grid cell overflow
                    continue;
                }

                int write_idx = cell_idx * grid.max_spheres_per_cell + slot;
                grid.cell_sphere_ids[write_idx] = sphere_idx;

                atomicAdd(grid.num_entries, 1);
            }
        }
    }
}

}//namespace ends

void allocate_gpu_spatial_grid(GpuSpatialGrid& grid, 
    float voxel_edge_length, 
    float cell_size,
    int max_spheres,
    float max_entries
){
    grid.cell_size = cell_size;
    grid.grid_dim_x = static_cast<int>(ceilf(voxel_edge_length / cell_size)); //    float voxel_edge_length = 50.0f;    float min_radius = 0.15f;
    //167
    grid.grid_dim_y = grid.grid_dim_x;
    grid.grid_dim_z = grid.grid_dim_x;
    grid.num_cells = grid.grid_dim_x * grid.grid_dim_y * grid.grid_dim_z; //167^3
    grid.max_spheres = max_spheres;
    grid.max_entries = max_entries;

    //tune later (Maybe add to params?)
    grid.max_spheres_per_cell = 256;

    const int bucket_entries = grid.num_cells * grid.max_spheres_per_cell; //1,185,185,185, so we need max_entries to be 1200000000
    if (bucket_entries > max_entries) {
        throw std::runtime_error("max entries too small");
    }
    CUDA_CHECK(cudaMalloc(&grid.cell_counts, grid.num_cells * sizeof(int)));
     std::cout << "for debugggggggging" << std::endl;

    CUDA_CHECK(cudaMalloc(&grid.cell_sphere_ids, bucket_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.num_entries, sizeof(int)));
    reset_gpu_spatial_grid(grid);
}

void free_gpu_spatial_grid(GpuSpatialGrid& grid) {//dealloc
    cudaFree(grid.cell_counts);
    cudaFree(grid.cell_sphere_ids);
    cudaFree(grid.num_entries);
    grid.cell_counts = nullptr;
    grid.cell_sphere_ids = nullptr;
    grid.num_entries = nullptr;
}

void reset_gpu_spatial_grid(GpuSpatialGrid& grid) {
    CUDA_CHECK(cudaMemset(grid.cell_counts, 0, grid.num_cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(grid.cell_sphere_ids,-1,grid.num_cells * grid.max_spheres_per_cell * sizeof(int)));
    CUDA_CHECK(cudaMemset(grid.num_entries, 0, sizeof(int)));
}

void insert_spheres_into_gpu_spatial_grid(
    GpuSimulationState& state,
    GpuSpatialGrid& grid,
    int begin_sphere,
    int end_sphere
) {
    if (end_sphere <= begin_sphere) {
        return;
    }

    int n = end_sphere - begin_sphere;
    int blocks = gpu_num_blocks(n);

    num_sphere_grid_entries_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(
        state,
        grid,
        begin_sphere,
        end_sphere
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void build_gpu_spatial_grid(
    GpuSimulationState& state,
    GpuSpatialGrid& grid
) {
    int sphere_count = 0;

    CUDA_CHECK(cudaMemcpy(
        &sphere_count,
        state.spheres.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    reset_gpu_spatial_grid(grid);

    if (sphere_count <= 0) {
        return;
    }

    insert_spheres_into_gpu_spatial_grid(state, grid, 0, sphere_count);
}