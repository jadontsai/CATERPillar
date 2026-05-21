#include "gpu_init.h"

#include <cuda_runtime.h>

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

__global__
//function makes just one front (pretty much a "smokier" smoke test)
void initialize_single_front_kernel(GpuSimulationState state) {
    //we only want one thread for now
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    float center = state.params.voxel_edge_length * 0.5f;
    //middle of the voxel, just for sanity checking
    float radius = state.params.min_radius;
    //both as given from GpuParameters

    //for future me, this is very intentionally a structure-of-arrays instead of an array of structs
    state.spheres.x[0] = center;
    state.spheres.y[0] = center;
    state.spheres.z[0] = center;
    state.spheres.r[0] = radius;

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
    state.fronts.r[0] = radius;
    //for now the same place as the sphere

    state.fronts.dir_x[0] = 1.0f;
    state.fronts.dir_y[0] = 0.0f;
    state.fronts.dir_z[0] = 0.0f;
    //defines initial growth direction, in this case +x

    state.fronts.object_type[0] = 0;
    state.fronts.object_id[0] = 0;
    state.fronts.branch_id[0] = 0;
    state.fronts.parent_sphere_id[0] = 0;
    state.fronts.active[0] = 1;
    //same as above

    *state.fronts.count = 1;
}

//this thing is cpu callable, launches the gpu kernel
void initialize_single_front_gpu(GpuSimulationState& state) {
    initialize_single_front_kernel<<<1, 1>>>(state);

    CUDA_CHECK(cudaGetLastError());
    //checks if the kernel launch actually worked
    CUDA_CHECK(cudaDeviceSynchronize());
   //wait for gpu to be done before cpu does anything (not neccesary later) 
}