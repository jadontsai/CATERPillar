#pragma once //only include header file once

struct GpuParameters {
    //different from original, just barebones for now
    //keep things as floats for now, no need for double
    float voxel_edge_length = 50.0f;
    float axons_icvf = 0.20f; //fraction not percent, double check this though
    float min_radius = 0.15f;
    float alpha = 4.0f;
    float beta = 0.25f;

    int max_spheres = 100000;//small for now
    int max_growth_fronts = 1000;
    int candidates_per_front = 256;
    int max_steps = 100;

    unsigned long long seed = 1;// apparently gpus like 64 bit seeds? anyways 1 is default
};