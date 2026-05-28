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


// namespace{


// }

// void generate_blood_vessels_kernel(GpuSimulationState state, int step) {
// //pseudocode (from: https://pmc.ncbi.nlm.nih.gov/articles/PMC8504684/table/T7/)
// // 1.	FOR i = 1 TO nTerminals DO
// // 2.	 terminal aT = venTerminalSampleGeneratorList.getSample //arterial terminal
// // 3.	 segments vS = choose closest two ∈ close_segment_list(venous_tree, constraintList)
// // 4.	 add_fork (venous_tree, aT, vS[0]) // with or without optimizing
// // 5.	 add_fork (venous_tree, aT, vS[1]) // with or without optimizing
// // 6.	 remove(venTerminalSampleGeneratorList, aT)
// // 7.	 balanceTree(venous_tree, d0) // update venous_tree diameter ratios
// // 8.	 terminal vT = artTerminalSampleGeneratorList.getSample //venous terminal
// // 9.	 segment aS = choose closest two ∈ close_segment_list(arterial_tree, constraintList)
// // 10.	 add_fork (arterial_tree, vT, aS[0]) // with or without optimizing
// // 11.	 add_fork (arterial_tree, vT, aS[1]) // with or without optimizing
// // 12.	 remove(artTerminalSampleGeneratorList, vT)
// // 13.	 balanceTree(artery_tree, d0) // update arterial_tree diameter ratios
// // 14.	ENDFOR
// }
// void generate_blood_vessels_gpu(GpuSimulationState state){

//     generate_blood_vessels_kernel<<<
// }
