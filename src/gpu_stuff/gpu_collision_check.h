#pragma once

#include "gpu_simulation_state.h"
#include "gpu_spatial_grid.h"

void run_collision_check(
    GpuSimulationState& state,
    GpuSpatialGrid& grid
);