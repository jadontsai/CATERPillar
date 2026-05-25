#pragma once

#include "gpu_simulation_state.h"

void initialize_single_front_gpu(GpuSimulationState& state);
void initialize_multiple_fronts_gpu(GpuSimulationState& state, int fronts);