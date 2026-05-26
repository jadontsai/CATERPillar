#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cmath>

__device__
float sample_standard_normal_box_muller(unsigned int seed) {
      //normally distributed radii for axon 
      // pertubation (like how much each sphere changes)
      float u1 = random_float(seed + 1u);
      float u2 = random_float(seed + 2u);

      u1 = fmaxf(u1, 1e-7f);

      float mag = sqrtf(-2.0f * logf(u1));
      float angle = 2.0f * 3.141592f * u2;

      return mag * cosf(angle);
}

__device__
float sample_normal_box_muller(
      unsigned int seed,
      float mean,
      float stddev
) {
      return mean + stddev * sample_standard_normal_box_muller(seed);
}
__device__//called from gpu, runs on gpu
unsigned int hash(unsigned int x) {
      //some psuedo random hashing function
      x = ((x >> 16) ^ x) * 0x6767f2a;
      x = ((x >> 16) ^ x) * 0x41d923b;
      x = (x >> 16) ^ x;
      return x;
}

__device__
float random_float(unsigned int seed) {
      //returns a random float between 0 and 1 based on the seed
      return static_cast<float>(hash(seed) &0x00FFFFFF)/16777216.0f;;
}
__device__
float safe_rsqrt(float x) {
      return rsqrtf(fmaxf(x, 1e-12f));
      //a bit safer behaviour than 0
}

__device__
void normalize3(float& x, float& y, float& z) {
      float inv_norm = safe_rsqrt(x * x + y * y + z * z);
      x *= inv_norm;
      y *= inv_norm;
      z *= inv_norm;
}

__device__
int clamp_int(int val, int min_val, int max_val) {
    return max(min(val, max_val), min_val);
}

__device__
int flatten_index(int x, int y, int z, const GpuSpatialGrid grid) {
    return x + grid.grid_dim_x * (y + z * grid.grid_dim_y);
}

__device__
float sample_gamma_integer_shape(unsigned int seed, int shape, float scale){
      float sum = 0.0f;
      for(int i = 0; i< shape; ++i){
            float u = random_float(seed + 1234u * static_cast<unsigned int>(i));
            u = fmaxf(u, 1e-7f);
            sum += -logf(u);
      }
      return scale *sum;
      }

      
__device__
void cell_bounds(
    float x, float y, float z, float r,
    const GpuSpatialGrid grid,
    int& min_x, int& max_x,
    int& min_y, int& max_y,
    int& min_z, int& max_z
)
{//a sphere occupies this range at maximum (overestimates a bit but it's close ish)
    min_x = static_cast<int>(floorf((x-r) / grid.cell_size));
    max_x = static_cast<int>(floorf((x+r) / grid.cell_size));
    min_y = static_cast<int>(floorf((y-r) / grid.cell_size));
    max_y = static_cast<int>(floorf((y+r) / grid.cell_size));
    min_z = static_cast<int>(floorf((z-r) / grid.cell_size));
    max_z = static_cast<int>(floorf((z+r) / grid.cell_size));

    //clamping to grid bounds near voxel boundaries
    min_x = clamp_int(min_x, 0, grid.grid_dim_x - 1);
    max_x = clamp_int(max_x, 0, grid.grid_dim_x - 1);
    min_y = clamp_int(min_y, 0, grid.grid_dim_y - 1);
    max_y = clamp_int(max_y, 0, grid.grid_dim_y - 1);
    min_z = clamp_int(min_z, 0, grid.grid_dim_z - 1);
    max_z = clamp_int(max_z, 0, grid.grid_dim_z - 1);
}