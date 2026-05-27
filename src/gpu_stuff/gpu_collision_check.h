#pragma once

#include "gpu_simulation_state.h"
#include "gpu_spatial_grid.h"

void run_collision_check(
    GpuSimulationState& state,
    GpuSpatialGrid& grid
);

void run_selected_candidate_conflict_check(//for candidate-candidate deconflict
    GpuSimulationState& state
);