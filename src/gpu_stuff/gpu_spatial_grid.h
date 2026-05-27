#pragma once

#include "gpu_simulation_state.h"

struct GpuSpatialGrid{
    float cell_size = 1.0f;
    //cells per voxel in each dimension
    int grid_dim_x = 0;
    int grid_dim_y = 0;
    int grid_dim_z = 0;
    //total cells is the product of the three numbers above
    int num_cells = 0;
    //maximum spheres in cell, for now just a safety thing,
    // don't think there's a physical reason
    int max_spheres = 0;
    //more than max spheres because one sphere can be in more than
    //one cell
    int max_entries = 0;

    // Fixed bucket capacity per cell.
    int max_spheres_per_cell = 0;

    // cell_counts[cell] = number of sphere ids currently stored in that cell.
    int* cell_counts = nullptr;

    // cell_sphere_ids[cell * max_spheres_per_cell + slot] = sphere id.
    int* cell_sphere_ids = nullptr;

    // Total number of cell/sphere references inserted.
    int* num_entries = nullptr;

};

void allocate_gpu_spatial_grid(GpuSpatialGrid& grid, 
    float voxel_edge_length, 
    float cell_size,
    int max_spheres,
    int max_entries
);

void free_gpu_spatial_grid(GpuSpatialGrid& grid);
void reset_gpu_spatial_grid(GpuSpatialGrid& grid);

void insert_spheres_into_gpu_spatial_grid(
    GpuSimulationState& state,
    GpuSpatialGrid& grid,
    int begin_sphere,
    int end_sphere
);

void build_gpu_spatial_grid(GpuSimulationState& state, 
    GpuSpatialGrid& grid);