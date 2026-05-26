#pragma once

#include "gpu_parameters.h"//uses the struct we just made

struct GpuSphereTable {
    //flattening the original c++ object for gpu memory access
    //these are all pointers to the flat arrays
    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    float* r = nullptr;
    float* base_r = nullptr; //for more realistic growth



    int* object_type = nullptr;//some number for some cell type, 0 for glial, etc etc
    int* object_id = nullptr;//which object (like axon 23) a sphere belongs to
    int* branch_id = nullptr;//which branch (only for glial cells)
    int* parent_sphere_id = nullptr;//for reconstruction

    int* count = nullptr;//number of spheres accepted
    int capacity = 0;//max spheres
};

struct GpuGrowthFrontTable {
    //"growth fronts" are where other spheres grow from
    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    float* r = nullptr;
    float* base_r = nullptr;


    float* dir_x = nullptr;
    float* dir_y = nullptr;
    float* dir_z = nullptr;

    int* object_type = nullptr;
    int* object_id = nullptr;
    int* branch_id = nullptr;
    int* parent_sphere_id = nullptr;

    int* active = nullptr;//stopped or not
    int* count = nullptr;
    int capacity = 0;
};

struct GpuCandidateTable {
    //temporary scratch space for candidate
    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    float* r = nullptr;

    float* dir_x = nullptr;
    float* dir_y = nullptr;
    float* dir_z = nullptr;

    int* front_id = nullptr;//which candidate came from which front
    int* parent_id = nullptr; //for collision detection (it's allowed to overlap with it's parent)
    int* valid = nullptr;
    int* selected = nullptr;//self explanatory

    int candidates_per_front = 0;
    int total_candidates = 0;
};

struct GpuSimulationState {
    GpuParameters params;

    GpuSphereTable spheres;
    GpuGrowthFrontTable fronts;
    GpuCandidateTable candidates;

    int* done = nullptr;//simulation done flag
    int* error_code = nullptr;//error code for debugging, since GPUs don't throw c++ exceptions apparently
};

void allocate_gpu_state(GpuSimulationState& state, const GpuParameters& params);
void free_gpu_state(GpuSimulationState& state);