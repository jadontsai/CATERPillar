#pragma once //only include header file once
//
struct GpuParameters {
    //different from original, just barebones for now
    //keep things as floats for now, no need for double
    float voxel_edge_length = 50.0f;
    float axons_icvf = 0.20f; //fraction not percent, double check this though
    float min_radius = 0.15f;
    float alpha = 4.0f;
    float beta = 0.25f;
    float overlap_factor = 4.0f;

    int max_spheres = 10000000;//small for now (not small anymore)
    int max_growth_fronts = 10000;
    int candidates_per_front = 256;
    int max_steps = 10000;

    unsigned long long seed = 1;// apparently gpus like 64 bit seeds? anyways 1 is default

    //TEMP FOR DEBUGGING
    int num_axons = 64;

    int num_glial_somas = 0;
    float glial_soma_radius_mean = 6.0f;
    float glial_soma_radius_std = 0.5f;
    int glial_primary_processes = 4;
    float glial_process_radius_fraction = 0.25f;
    float glial_process_persistence = 0.70f;

    int num_blood_vessels = 0;
    float blood_vessel_radius = 2.0f;
    float blood_vessel_persistence = 0.95f;
};