#pragma once
#include "gpu_simulation_state.h"
#include "gpu_object_types.h"

float sample_standard_normal_box_muller(unsigned int seed);
void cell_bounds(
    float x, float y, float z, float r,
    const GpuSpatialGrid grid,
    int& min_x, int& max_x,
    int& min_y, int& max_y,
    int& min_z, int& max_z
);
float sample_gamma_integer_shape(unsigned int seed, int shape, float scale);
int flatten_index(int x, int y, int z, const GpuSpatialGrid grid);
int clamp_int(int val, int min_val, int max_val);
void normalize3(float& x, float& y, float& z);
float safe_rsqrt(float x) ;
float random_float(unsigned int seed);
unsigned int hash(unsigned int x);
float sample_normal_box_muller(
      unsigned int seed,
      float mean,
      float stddev
);
