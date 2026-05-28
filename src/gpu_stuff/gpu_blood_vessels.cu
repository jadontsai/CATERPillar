#include "gpu_blood_vessels.h"
#include "gpu_util.cuh"
#include "gpu_object_types.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cmath>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err));    \
        }                                                                  \
    } while (0)


namespace{


}

void generate_blood_vessels_gpu(GpuSimulationState state){

    
}
