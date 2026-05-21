#include "gpu_runner.h"
#include "gpu_simulation_state.h"
#include "gpu_collision_check.h"
#include "gpu_init.h"

#include <cuda_runtime.h>
#include <iostream>
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

__global__//launched from cpu, runs on gpu
void gpu_smoke_test_kernel(int* error_code) {
    //threadIdx.x is the thread index within the block, blockIdx.x is the block index within the grid
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *error_code = 1234;
        //if it prints this, then this worked
    }
}

void run_gpu_simulation(const GpuParameters& params) {
    std::cout << "doing gpu stuff..." << std::endl;

    GpuSimulationState state;
    allocate_gpu_state(state, params);
    //so state is on the cpu stack, but it's pointing to GPU memory

    // gpu_smoke_test_kernel<<<1, 32>>>(state.error_code);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // int host_error_code = 0;

    // CUDA_CHECK(cudaMemcpy(
    //     &host_error_code,
    //     state.error_code,
    //     sizeof(int),
    //     cudaMemcpyDeviceToHost
    // ));

    // std::cout << "should be 1234 if it worked: " << host_error_code << std::endl;
    initialize_single_front_gpu(state);
    //launches the single front kernel with 1 block and 1 thread

    //cpu variables that will be copied back from gpu
    int sphere_count =0;
    int front_count = 0;

    float x_0 = 0.0f;
    float y_0 = 0.0f;
    float z_0 = 0.0f;
    float r_0 = 0.0f;

    //actually copying things back
    CUDA_CHECK(cudaMemcpy(
        &sphere_count,
        state.spheres.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    //expected value of 1, since there's 1 sphere
    CUDA_CHECK(cudaMemcpy(
        &front_count,
        state.fronts.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    //expected value of 1, since there's 1 front
    CUDA_CHECK(cudaMemcpy(
        &x_0,
        state.spheres.x,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        &y_0,
        state.spheres.y,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        &z_0,
        state.spheres.z,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        &r_0,
        state.spheres.r,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    std::cout << "sphere count (should be 1): " << sphere_count << std::endl;
    std::cout << "front count (should be 1): " << front_count << std::endl;
    std::cout << "sphere 0 position (should be half voxel edge length): " << x_0 << ", " << y_0 << ", " << z_0 << std::endl;
    std::cout << "sphere 0 radius (should be min radius): " << r_0 << std::endl;

    free_gpu_state(state);
}