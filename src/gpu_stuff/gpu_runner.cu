#include "gpu_runner.h"
#include "gpu_simulation_state.h"
#include "gpu_collision_check.h"
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

    gpu_smoke_test_kernel<<<1, 32>>>(state.error_code);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int host_error_code = 0;

    CUDA_CHECK(cudaMemcpy(
        &host_error_code,
        state.error_code,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    std::cout << "should be 1234 if it worked: " << host_error_code << std::endl;

    free_gpu_state(state);
}