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
    //how many cells a sphere touches
    int* sphere_touching_num = nullptr;
    //for indexing
    int* sphere_touching_num_offset = nullptr;
    int* grid_cell_id = nullptr;//which cell
    int* grid_sphere_id = nullptr;

    //two pointers to find range of entries for the 
    // cell; if start is -1 then no spheres in cell
    // apparently this is the most efficient way to do this
    int* cell_start = nullptr;
    int* cell_end = nullptr;

    //GPU side variable to store number of valid entries in build (so it doesn't 
    // cross max entries
    int* num_entries = nullptr;
};

void allocate_gpu_spatial_grid(GpuSpatialGrid& grid, 
    float voxel_edge_length, 
    float cell_size,
    int max_spheres,
    int max_entries,
);

void free_gpu_spatial_grid(GpuSpatialGrid& grid);

void build_gpu_spatial_grid(const GpuSimulationState& state, 
    GpuSpatialGrid& grid);