#pragma once
#include "gpu_simulation_state.h"

void commit_candidates_gpu(GpuSimulationState& state);

void commit_candidates_and_update_grid_gpu(
    GpuSimulationState& state,
    GpuSpatialGrid& grid
);