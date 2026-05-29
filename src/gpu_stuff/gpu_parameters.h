#pragma once
struct GpuParameters {
    //different from original, just barebones for now
    //keep things as floats for now, no need for double
    float voxel_edge_length = 50.0f;
    float axons_icvf = 0.20f; //fraction not percent, double check this though
    float min_radius = 0.15f;
    float alpha = 4.0f;
    float beta = 0.25f;
    float overlap_factor = 4.0f;
    float beading = 0.05f;
    float persistence = 0.9f;
    float bounds = 0.2f;

    //bv
    float bv_norm_dist = 0.05f;
    float bv_norm_norm =0.5f; //probably need a better name
    float z_bottom = 0.05f;
    float z_top = 0.95f;

    int max_spheres = 10000000;//small for now (not small anymore)// cap seems to be 1000000000 (too big)
    float max_entries = 1200000000.0f;
    int max_growth_fronts = 1000;
    int candidates_per_front = 1024;
    int max_steps = 100;
    int runs = 100000;

    //geometries
    int grid_width =15; //initial placement width (total is this squared)
    float spacing = 5.0f; // min spacing (wait this is overconstrained)

    unsigned long long seed = 1;// apparently gpus like 64 bit seeds? anyways 1 is default

    //TEMP FOR DEBUGGING
    int num_axons = 225;
    int num_glial_somas = 0;
    int num_pial_arteries = 4;
    int num_pial_veins = 2;
    int num_diving_arteries = 8;
    int num_ascending_veins = 4;
    float artery_to_vein_ratio = 2.1f;//from paper, this is a value
    // MAB: fewer branches off diving arteries than ascending veins.
    int artery_branches_per_diving_vessel = 2;
    int vein_branches_per_ascending_vessel = 5;
    
    //relative radii
    float glial_soma_radius_mean = 6.0f;
    float glial_soma_radius_std = 0.5f;
    int glial_primary_processes = 4;
    float glial_process_radius_fraction = 0.25f;
    float glial_process_persistence = 0.70f;
    //blood
    float pial_artery_radius = 3.0f;
    float pial_vein_radius = 4.0f;
    float penetrating_artery_radius = 1.5f;
    float penetrating_vein_radius = 2.0f;
    float capillary_radius = 0.4f;
    float vessel_persistence = 0.97f;//should be very big need some tortuosity still though
    float capillary_persistence = 0.50f;
    int pial_vessel_steps = 40;
    int penetrating_vessel_steps = 80;
    int capillary_steps = 20;
};