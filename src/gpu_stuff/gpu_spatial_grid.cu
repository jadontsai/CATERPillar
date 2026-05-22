#include "gpu_spatial_grid.h"
#include "gpu_launch_config.h"
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

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err));    \
        }                                                                  \
    } while (0)

namespace{//makes things private

__device__
int clamp_int(int val, int min_val, int max_val) {
    return max(min(val, max_val), min_val);
}

__device__
int flatten_index(int x, int y, int z, const GpuSpatialGrid grid) {
    return x + grid.grid_dim_x * (y + z * grid.grid_dim_y);
}//converts 3d cell coordinates to 1d index for sorting

__device__
void cell_bounds(
    float x, float y, float z, float r,
    const GpuSpatialGrid grid,
    int& min_x, int& max_x,
    int& min_y, int& max_y,
    int& min_z, int& max_z
)
{//a sphere occupies this range at maximum (overestimates a bit but it's close ish)
    min_x = static_cast<int>(floorf((x-r) / grid.cell_size));
    max_x = static_cast<int>(floorf((x+r) / grid.cell_size));
    min_y = static_cast<int>(floorf((y-r) / grid.cell_size));
    max_y = static_cast<int>(floorf((y+r) / grid.cell_size));
    min_z = static_cast<int>(floorf((z-r) / grid.cell_size));
    max_z = static_cast<int>(floorf((z+r) / grid.cell_size));

    //clamping to grid bounds near voxel boundaries
    min_x = clamp_int(min_x, 0, grid.grid_dim_x - 1);
    max_x = clamp_int(max_x, 0, grid.grid_dim_x - 1);
    min_y = clamp_int(min_y, 0, grid.grid_dim_y - 1);
    max_y = clamp_int(max_y, 0, grid.grid_dim_y - 1);
    min_z = clamp_int(min_z, 0, grid.grid_dim_z - 1);
    max_z = clamp_int(max_z, 0, grid.grid_dim_z - 1);
}

__global__
void num_sphere_grid_entries_gpu(
    GpuSimulationState state,
    GpuSpatialGrid grid
)   
{//how many grid cells does a sphere touch
    int sphere_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sphere_count = *state.spheres.count;

    //avoid out of bounds access, so excess threads exit early
    if (sphere_idx >= sphere_count) return;

    int x_min, x_max, y_min, y_max, z_min, z_max;


    cell_bounds(
        state.spheres.x[sphere_idx],
        state.spheres.y[sphere_idx],
        state.spheres.z[sphere_idx],
        state.spheres.r[sphere_idx],
        grid,
        x_min, x_max, y_min, y_max, z_min, z_max
    );

    //number of cells touched is the product of the ranges in each dimension, so if 
    //say a sphere ranges from x=56 to x=58, that's 3 cells in the x dimension (56, 57, 58)
    // multiply that by the number of cells in the y and z dimensions to get total number of cells touched
    int entries_for_sphere = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1);

    //stores the count of entries for each sphere, which 
    // will be used to compute offsets for where to write
    // the actual entries in the next kernel
    grid.sphere_grid_entry_counts[sphere_idx] = entries_for_sphere;
}

__global__
void fill_sphere_grid_entries_gpu(
    GpuSimulationState state,
    GpuSpatialGrid grid
)   
{
    //same as usual
    int sphere_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sphere_count = *state.spheres.count;

    if (sphere_idx >= sphere_count) return;

    int x_min, x_max, y_min, y_max, z_min, z_max;
    cell_bounds(
        state.spheres.x[sphere_idx],
        state.spheres.y[sphere_idx],
        state.spheres.z[sphere_idx],
        state.spheres.r[sphere_idx],
        grid,
        x_min, x_max, y_min, y_max, z_min, z_max
    );

    //uses prefix sum of entry counts to find the starting index in 
    // the grid entries array where this sphere's
    //  entries should be written
    int entry_start = grid.sphere_grid_entry_offsets[sphere_idx];
    int entry_idx = 0;

    for (int z = z_min; z <= z_max; ++z) {
        for (int y = y_min; y <= y_max; ++y) {
            for (int x = x_min; x <= x_max; ++x) {
                int flat_cell_idx = flatten_index(x, y, z, grid);
                grid.sphere_grid_entries[entry_start + entry_idx] = flat_cell_idx;
                ++entry_idx;
            }
        }
    }
}

__global__
void create_cell_range(
    GpuSimulationState state,
    GpuSpatialGrid grid
){//runs after sorting, creates starting and ending indices/pointers
    int entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_entries = grid.total_sphere_grid_entries;
    //if out of bounds
    if (entry_idx >= total_entries) return;

    int cell_idx = grid.sphere_grid_entries[entry_idx];

    if(entry_idx == 0){
        grid.cell_range_start[cell_idx] = 0;
    }
    else{
        int prev_cell_idx = grid.sphere_grid_entries[entry_idx - 1];
        if(cell_idx != prev_cell_idx){
            //transition between cells, so put the pointers here
            grid.cell_range_start[cell_idx] = entry_idx;
            grid.cell_range_end[prev_cell_idx] = entry_idx;
        }
    }
    if(entry_idx == total_entries - 1){
        grid.cell_range_end[cell_idx] = total_entries;
    }
}
}//namespace ends


__global__
void allocate_spatial_grid_gpu(GpuSpatialGrid* grid,
    float cell_size,
    float voxel_edge_length,
    int max_spheres, 
    int max_entries) {
    grid.cell_size = cell_size;

    grid.grid_dim_x = static_cast<int>(ceilf(voxel_edge_length / cell_size));
    grid.grid_dim_y = grid.grid_dim_x;
    grid.grid_dim_z = grid.grid_dim_x;

    int total_cells = grid.grid_dim_x * grid.grid_dim_y * grid.grid_dim_z;

    grid.max_spheres = max_spheres;
    grid.total_sphere_grid_entries = max_entries;

    //allocating arrays
    CUDA_CHECK(cudaMalloc(&grid.sphere_grid_entry_counts, max_spheres * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.sphere_grid_entry_offsets, max_spheres * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&grid.cell_start, grid.total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.cell_end, grid.total_cells * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&grid.grid_cell_id, max_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.grid_sphere_id, max_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.num_entries,  sizeof(int)));
    CUDA_CHECK(cudaMemset(grid.num_entries, 0, sizeof(int)));
}

void free_spatial_grid_gpu(GpuSpatialGrid* grid) {
    CUDA_CHECK(cudaFree(grid.sphere_grid_entry_counts));
    CUDA_CHECK(cudaFree(grid.sphere_grid_entry_offsets));
    CUDA_CHECK(cudaFree(grid.cell_start));
    CUDA_CHECK(cudaFree(grid.cell_end));
    CUDA_CHECK(cudaFree(grid.grid_cell_id));
    CUDA_CHECK(cudaFree(grid.grid_sphere_id));
    CUDA_CHECK(cudaFree(grid.num_entries));

    //to avoid accidental use
    grid.sphere_grid_entry_counts = nullptr;
    grid.sphere_grid_entry_offsets = nullptr;
    grid.cell_start = nullptr;      
    grid.cell_end = nullptr;
    grid.grid_cell_id = nullptr;
    grid.grid_sphere_id = nullptr;
    grid.num_entries = nullptr;
}

void build_spatial_grid_gpu(GpuSimulationState state, GpuSpatialGrid grid) {
    int sphere_count = 0;
    //thrust wants need count to be on host, so we have
    //to copy it there
    CUDA_CHECK(cudaMemcpy(&sphere_count,
         state.spheres.count, sizeof(int),
         cudaMemcpyDeviceToHost));

    //if no sphere, then no entries
    if (sphere_count <=0){
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(grid.num_entries, 
            &zero, sizeof(int), cudaMemcpyHostToDevice));
        return;
    }

    int spheres_per_block = gpu_max_threads_per_block(sphere_count);
    //launches one thread per sphere
    count_sphere_grid_entries_gpu<<<spheres_per_block, GPU_THREADS_PER_BLOCK>>>
    (state, grid);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    //prefix sum, wrapping pointers
    thrust::device_ptr<int> entry_counts_ptr(grid.sphere_grid_entry_counts);
    thrust::device_ptr<int> entry_offsets_ptr(grid.sphere_grid_entry_offsets);
    //actually computing the sum
    thrust::exclusive_scan(entry_counts_ptr, 
        entry_counts_ptr + sphere_count, 
        entry_offsets_ptr);

    int last_count = 0;
    int last_offset = 0;
    CUDA_CHECK(cudaMemcpy(&last_count, grid.sphere_grid_entry_counts + sphere_count - 1,
         sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_offset, grid.sphere_grid_entry_offsets + sphere_count - 1,
         sizeof(int), cudaMemcpyDeviceToHost));

    int total_entries = last_offset + last_count;

    if (total_entries > grid.total_sphere_grid_entries) {
        throw std::runtime_error("Need bigger max entries");
    }

    //kernel needs total entries
    CUDA_CHECK(cudaMemcpy(grid.total_entries, 
    &total_entries, sizeof(int), cudaMemcpyHostToDevice));

    fill_sphere_grid_entries_gpu<<<blocks_for_spheres, threads_per_block>>>(state, grid);
   
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    
    thrust::device_ptr<int> grid_cell_id_ptr(grid.grid_cell_id);
    thrust::device_ptr<int> grid_sphere_id_ptr(grid.grid_sphere_id);

    thrust::sort_by_key(
        grid_cell_id_ptr,
        grid_cell_id_ptr + total_entries,
        grid_sphere_id_ptr
    );

    CUDA_CHECK(cudaMemset(grid.cell_start, -1, grid.total_cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(grid.cell_end, -1, grid.total_cells * sizeof(int)));

    int threads_per_block = gpu_max_threads_per_block(total_entries);
    build_cell_range<<<blocks_for_entries, threads_per_block>>>(state, grid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}