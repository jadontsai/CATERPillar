//this file initialiezs either one axon (single front) or a lot of axons
// (multiple fronts); placement is done here, as well as initial radius size
// Radius size follows a rough gamma distribution (can only take integers for now)
// Gamma distribution is a sum of two exponentials (Hence why only integer inputs)



#include "gpu_init.h"
#include "gpu_util.h"
#include "gpu_object_types.h"

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

    //float r = state.params.min_radius;
    int shape = static_cast<int>(roundf(state.params.alpha));
    shape = max(shape, 1);//shape of the gamma distribution...

    unsigned int seed =
        static_cast<unsigned int>(
            state.params.seed +
            5678ULL * static_cast<unsigned long long>(id)
        );//some random value

    float base_radius = sample_gamma_integer_shape(
        seed + 50000u,
        shape,
        state.params.beta
    );

    base_radius = fmaxf(base_radius, state.params.min_radius);
    float r = base_radius;
    //initial sphere
    state.spheres.x[id] = x;
    state.spheres.y[id] = y;
    state.spheres.z[id] = z;
    state.spheres.r[id] = r;
    state.spheres.base_r[id] = base_radius;
    state.spheres.object_type[id] = 0;
    state.spheres.object_id[id] = id;
    state.spheres.branch_id[id] = 0;
    state.spheres.parent_sphere_id[id] = -1;

    state.fronts.x[id] = x;
    state.fronts.y[id] = y;
    state.fronts.z[id] = z;
    state.fronts.r[id] = r;
    state.fronts.base_r[id] = base_radius;

    state.fronts.dir_x[id] = 0.0f;
    state.fronts.dir_y[id] = 0.0f;
    state.fronts.dir_z[id] = 1.0f;

    state.fronts.object_type[id] = 0;
    state.fronts.object_id[id] = id;
    state.fronts.branch_id[id] = 0;
    state.fronts.parent_sphere_id[id] = id;
    state.fronts.active[id] = 1;
}
//helpers
__device__
float sample_axon_base_radius(
    GpuSimulationState state,
    int object_id
) {
    int shape = static_cast<int>(roundf(state.params.alpha));
    shape = max(shape, 1);

    unsigned int seed = static_cast<unsigned int>(
        state.params.seed +
        7919ULL * static_cast<unsigned long long>(object_id)
    );

    float radius = sample_gamma_integer_shape(
        seed + 50000u,
        shape,
        state.params.beta
    );

    radius = fmaxf(radius, state.params.min_radius);

    return radius;
}
__device__
float sample_glial_soma_radius(
    GpuSimulationState state,
    int soma_id
) {
    unsigned int seed = static_cast<unsigned int>(
        state.params.seed +
        104729ULL * static_cast<unsigned long long>(soma_id)
    );

    float radius = sample_normal_box_muller(
        seed + 70000u,
        state.params.glial_soma_radius_mean,
        state.params.glial_soma_radius_std
    );

    radius = fmaxf(radius, state.params.min_radius);

    return radius;
}


__global__
void initialize_many_axon_fronts_kernel(
    GpuSimulationState state,
    int num_axons
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= num_axons) {
        return;
    }

    if (id >= state.spheres.capacity || id >= state.fronts.capacity) {
        *state.error_code = 1;
        return;
    }

    float voxel_edge = state.params.voxel_edge_length;
    int grid_width = 8;
    int ix = id % grid_width;
    int iy = id / grid_width;
    float spacing = 5.0f;
    float x = spacing * static_cast<float>(ix + 1);
    float y = spacing * static_cast<float>(iy + 1);
    float z = 0.5f * voxel_edge;

    float base_radius = sample_axon_base_radius(state, id);
    float r = base_radius;

    state.spheres.x[id] = x;
    state.spheres.y[id] = y;
    state.spheres.z[id] = z;
    state.spheres.r[id] = r;
    state.spheres.base_r[id] = base_radius;

    state.spheres.object_type[id] = GPU_OBJECT_AXON;
    state.spheres.object_id[id] = id;
    state.spheres.branch_id[id] = 0;
    state.spheres.parent_sphere_id[id] = -1;

    state.fronts.x[id] = x;
    state.fronts.y[id] = y;
    state.fronts.z[id] = z;
    state.fronts.r[id] = r;
    state.fronts.base_r[id] = base_radius;

    state.fronts.dir_x[id] = 0.0f;
    state.fronts.dir_y[id] = 0.0f;
    state.fronts.dir_z[id] = 1.0f;

    state.fronts.object_type[id] = GPU_OBJECT_AXON;
    state.fronts.object_id[id] = id;
    state.fronts.branch_id[id] = 0;
    state.fronts.parent_sphere_id[id] = id;
    state.fronts.active[id] = 1;
}

__global__
void initialize_glial_somas_kernel(
    GpuSimulationState state,
    int start_sphere_id,
    int num_somas
) {
    int soma_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (soma_id >= num_somas) {
        return;
    }

    int sphere_id = start_sphere_id + soma_id;

    if (sphere_id >= state.spheres.capacity) {
        *state.error_code = 2;
        return;
    }

    float voxel_edge = state.params.voxel_edge_length;

    int grid_width = 8;

    int ix = soma_id % grid_width;
    int iy = soma_id / grid_width;

    float spacing = 8.0f;

    float x = 0.25f * voxel_edge + spacing * static_cast<float>(ix);
    float y = 0.25f * voxel_edge + spacing * static_cast<float>(iy);
    float z = 0.5f * voxel_edge;

    x = fminf(fmaxf(x, 0.0f), voxel_edge);
    y = fminf(fmaxf(y, 0.0f), voxel_edge);
    z = fminf(fmaxf(z, 0.0f), voxel_edge);

    float radius = sample_glial_soma_radius(state, soma_id);

    state.spheres.x[sphere_id] = x;
    state.spheres.y[sphere_id] = y;
    state.spheres.z[sphere_id] = z;
    state.spheres.r[sphere_id] = radius;
    state.spheres.base_r[sphere_id] = radius;

    state.spheres.object_type[sphere_id] = GPU_OBJECT_GLIAL_SOMA;
    state.spheres.object_id[sphere_id] = soma_id;
    state.spheres.branch_id[sphere_id] = 0;
    state.spheres.parent_sphere_id[sphere_id] = -1;
}

__global__
void initialize_glial_process_fronts_kernel(
    GpuSimulationState state,
    int soma_start_sphere_id,
    int front_start_id,
    int num_somas,
    int processes_per_soma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_processes = num_somas * processes_per_soma;

    if (idx >= total_processes) {
        return;
    }

    int soma_id = idx / processes_per_soma;
    int process_id = idx % processes_per_soma;

    int soma_sphere_id = soma_start_sphere_id + soma_id;
    int front_id = front_start_id + idx;

    if (front_id >= state.fronts.capacity) {
        *state.error_code = 3;
        return;
    }

    float sx = state.spheres.x[soma_sphere_id];
    float sy = state.spheres.y[soma_sphere_id];
    float sz = state.spheres.z[soma_sphere_id];
    float sr = state.spheres.r[soma_sphere_id];

    float angle =
        2.0f * 3.141592f *
        static_cast<float>(process_id) /
        static_cast<float>(processes_per_soma);

    float dx = cosf(angle);
    float dy = sinf(angle);
    float dz = 0.25f;

    normalize3(dx, dy, dz);

    float process_radius = fmaxf(
        state.params.glial_process_radius_fraction * sr,
        state.params.min_radius
    );

    state.fronts.x[front_id] = sx + sr * dx;
    state.fronts.y[front_id] = sy + sr * dy;
    state.fronts.z[front_id] = sz + sr * dz;
    state.fronts.r[front_id] = process_radius;
    state.fronts.base_r[front_id] = process_radius;

    state.fronts.dir_x[front_id] = dx;
    state.fronts.dir_y[front_id] = dy;
    state.fronts.dir_z[front_id] = dz;

    state.fronts.object_type[front_id] = GPU_OBJECT_GLIAL_PROCESS;
    state.fronts.object_id[front_id] = soma_id;
    state.fronts.branch_id[front_id] = process_id;
    state.fronts.parent_sphere_id[front_id] = soma_sphere_id;
    state.fronts.active[front_id] = 1;
}


//end of kernels

void initialize_glial_somas_gpu(
    GpuSimulationState& state,
    int start_sphere_id,
    int num_somas
) {
    if (num_somas <= 0) {
        return;
    }

    if (start_sphere_id < 0 ||
        start_sphere_id + num_somas > state.spheres.capacity) {
        throw std::runtime_error("Too many glial soma spheres");
    }

    int blocks = gpu_num_blocks(num_somas);

    initialize_glial_somas_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(
        state,
        start_sphere_id,
        num_somas
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void initialize_glial_process_fronts_gpu(
    GpuSimulationState& state,
    int soma_start_sphere_id,
    int front_start_id,
    int num_somas,
    int processes_per_soma
) {
    if (num_somas <= 0 || processes_per_soma <= 0) {
        return;
    }

    int total_processes = num_somas * processes_per_soma;

    if (front_start_id < 0 ||
        front_start_id + total_processes > state.fronts.capacity) {
        throw std::runtime_error("Too many glial process fronts");
    }

    int blocks = gpu_num_blocks(total_processes);

    initialize_glial_process_fronts_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(
        state,
        soma_start_sphere_id,
        front_start_id,
        num_somas,
        processes_per_soma
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void initialize_scene_gpu(GpuSimulationState& state) {
    int num_axons = state.params.num_axons;
    int num_glial_somas = state.params.num_glial_somas;
    int processes_per_soma = state.params.glial_primary_processes;

    if (num_axons < 0 || num_glial_somas < 0 || processes_per_soma < 0) {
        throw std::runtime_error("Invalid scene initialization counts");
    }

    int soma_start_sphere_id = num_axons;
    int num_soma_spheres = num_glial_somas;

    int axon_front_start = 0;
    int glial_process_front_start = num_axons;

    int num_glial_process_fronts = num_glial_somas * processes_per_soma;

    int total_initial_spheres = num_axons + num_soma_spheres;
    int total_initial_fronts = num_axons + num_glial_process_fronts;

    if (total_initial_spheres > state.spheres.capacity) {
        throw std::runtime_error("Not enough sphere capacity for initial scene");
    }

    if (total_initial_fronts > state.fronts.capacity) {
        throw std::runtime_error("Not enough front capacity for initial scene");
    }

    if (num_axons > 0) {
        int blocks = gpu_num_blocks(num_axons);

        initialize_many_axon_fronts_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(
            state,
            num_axons
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    initialize_glial_somas_gpu(
        state,
        soma_start_sphere_id,
        num_glial_somas
    );

    initialize_glial_process_fronts_gpu(
        state,
        soma_start_sphere_id,
        glial_process_front_start,
        num_glial_somas,
        processes_per_soma
    );

    CUDA_CHECK(cudaMemcpy(
        state.spheres.count,
        &total_initial_spheres,
        sizeof(int),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        state.fronts.count,
        &total_initial_fronts,
        sizeof(int),
        cudaMemcpyHostToDevice
    ));

    (void)axon_front_start;
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