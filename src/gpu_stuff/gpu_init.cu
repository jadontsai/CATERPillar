//this file initialiezs either one axon (single front) or a lot of axons
// (multiple fronts); placement is done here, as well as initial radius size
// Radius size follows a rough gamma distribution (can only take integers for now)
// Gamma distribution is a sum of two exponentials (Hence why only integer inputs)



#include "gpu_init.h"

#include <cuda_runtime.h>
#include "gpu_launch_config.h"//for num blocks
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err));    \
        }                                                                  \
    } while (0)
//i am being very lazy and duplicating functions rn
//maybe one day these get put into their own file
namespace {
__device__

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
float sample_gamma_integer_shape(unsigned int seed, int shape, float scale){
      float sum = 0.0f;
      for(int i = 0; i< shape; ++i){
            float u = random_float(seed + 1234u * static_cast<unsigned int>(i));
            u = fmaxf(u, 1e-7f);
            sum += -logf(u);
      }
      return scale *sum;
      }
    }
__global__
//function makes just one front (pretty much a "smokier" smoke test)
void initialize_single_front_kernel(GpuSimulationState state) {
    //we only want one thread for now
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    float center = state.params.voxel_edge_length * 0.5f;
    //middle of the voxel, just for sanity checking

    //making radius gamma distributed
    int shape = static_cast<int>(roundf(state.params.alpha));
    shape = max(shape, 1);

    float initial_radius = sample_gamma_integer_shape(static_cast<unsigned int>(state.params.seed + 999u),
        shape,
        state.params.beta
    );

    initial_radius = fmaxf(initial_radius, state.params.min_radius);

    state.spheres.r[0] = initial_radius;
    //front radius is same as original radius for now
    state.fronts.r[0] = initial_radius;

    //for future me, this is very intentionally a structure-of-arrays instead of an array of structs
    state.spheres.x[0] = center;
    state.spheres.y[0] = center;
    state.spheres.z[0] = center;

    state.spheres.object_type[0] = 0;     
    //object type is the type of cell, and we define 0 as axon
    state.spheres.object_id[0] = 0;
    // object ID is axon 0, 1, 2 etc
    state.spheres.branch_id[0] = 0;
    
    state.spheres.parent_sphere_id[0] = -1;
    //we defined -1 to mean no parent (it is the root)
    *state.spheres.count = 1;
    //rmr we have to derefence this because it's a pointer 

    state.fronts.x[0] = center;
    state.fronts.y[0] = center;
    state.fronts.z[0] = center;
    //for now the same place as the sphere

    state.fronts.dir_x[0] = 0.0f;
    state.fronts.dir_y[0] = 0.0f;
    state.fronts.dir_z[0] = 1.0f;
    //defines initial growth direction, in this case +x
    //nvm thye used +z, can change though

    state.fronts.object_type[0] = 0;
    state.fronts.object_id[0] = 0;
    state.fronts.branch_id[0] = 0;
    state.fronts.parent_sphere_id[0] = 0;
    state.fronts.active[0] = 1;
    //same as above

    *state.fronts.count = 1;
}
__global__
void initialize_multiple_fronts_kernel(GpuSimulationState state, int num_fronts){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_fronts||id >= state.spheres.capacity || id >= state.fronts.capacity) {
        return;
    }

    float voxel_edge = state.params.voxel_edge_length;

    //temporary deterministic placement to spread around fronts
    int grid_width = 8;

    int ix = id % grid_width;
    int iy = id / grid_width;

    float spacing = 5.0f;  // temp

    float center = 0.5f * voxel_edge;

    float x = center + spacing * static_cast<float>(ix - grid_width / 2);
    float y = center + spacing * static_cast<float>(iy - grid_width / 2);
    float z = center;

    float r = state.params.min_radius;

    //Initial committed sphere for this front
    state.spheres.x[id] = x;
    state.spheres.y[id] = y;
    state.spheres.z[id] = z;
    state.spheres.r[id] = r;

    state.spheres.object_type[id] = 0;
    state.spheres.object_id[id] = id;
    state.spheres.branch_id[id] = 0;
    state.spheres.parent_sphere_id[id] = -1;

    state.fronts.x[id] = x;
    state.fronts.y[id] = y;
    state.fronts.z[id] = z;
    state.fronts.r[id] = r;

    state.fronts.dir_x[id] = 0.0f;
    state.fronts.dir_y[id] = 0.0f;
    state.fronts.dir_z[id] = 1.0f;

    state.fronts.object_type[id] = 0;
    state.fronts.object_id[id] = id;
    state.fronts.branch_id[id] = 0;
    state.fronts.parent_sphere_id[id] = id;
    state.fronts.active[id] = 1;
}
void initialize_multiple_fronts_gpu(GpuSimulationState& state, int num_fronts){
//very test version
    if (num_fronts <= 0){
        return;
    }

    if (num_fronts > state.spheres.capacity || num_fronts > state.fronts.capacity){
        throw std::runtime_error("Too many initial fronts");
    }
    int blocks = gpu_num_blocks(num_fronts);

    initialize_multiple_fronts_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(state, num_fronts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(
        state.spheres.count,
        &num_fronts,
        sizeof(int),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        state.fronts.count,
        &num_fronts,
        sizeof(int),
        cudaMemcpyHostToDevice
    ));
}



//this thing is cpu callable, launches the gpu kernel
void initialize_single_front_gpu(GpuSimulationState& state) {
    initialize_single_front_kernel<<<1, 1>>>(state);

    CUDA_CHECK(cudaGetLastError());
    //checks if the kernel launch actually worked
    CUDA_CHECK(cudaDeviceSynchronize());
   //wait for gpu to be done before cpu does anything (not neccesary later) 
}