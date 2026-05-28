#pragma once

#include "gpu_simulation_state.h"
#include "gpu_parameters.h"

void initialize_single_front_gpu(GpuSimulationState& state);
void initialize_multiple_fronts_gpu(GpuSimulationState& state, int fronts);
void initialize_scene_gpu(GpuSimulationState& state);
void initialize_glial_somas_gpu(
    GpuSimulationState& state,
    int start_sphere_id,
    int num_somas
);

void initialize_glial_process_fronts_gpu(
    GpuSimulationState& state,
    int soma_start_sphere_id,
    int front_start_id,
    int num_somas,
    int processes_per_soma
);
