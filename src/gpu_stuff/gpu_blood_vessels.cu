#include "gpu_blood_vessels.h"
#include "gpu_util.cuh"
#include "gpu_launch_config.h"
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


namespace {

__device__ inline
void write_sphere(//this logic exists somewhere else but i dont wanna have a mess of a helper file
    GpuSimulationState state,
    int sphere_id,
    float x,
    float y,
    float z,
    float r,
    int object_type,
    int object_id,
    int branch_id,
    int parent_sphere_id
) {
    state.spheres.x[sphere_id] = x;
    state.spheres.y[sphere_id] = y;
    state.spheres.z[sphere_id] = z;
    state.spheres.r[sphere_id] = r;
    state.spheres.base_r[sphere_id] = r;

    state.spheres.object_type[sphere_id] = object_type;
    state.spheres.object_id[sphere_id] = object_id;
    state.spheres.branch_id[sphere_id] = branch_id;
    state.spheres.parent_sphere_id[sphere_id] = parent_sphere_id;
}

__global__
void generate_blood_vessels_kernel(
    GpuSimulationState state,
    int start_sphere_id
) {
    int vessel_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_vessels = state.params.num_pial_arteries;//for now

    if (vessel_id >= num_vessels) {
        return;
    }

    // First pass, one chain per vessel
    int object_type = (vessel_id % 2 == 0)? GPU_OBJECT_ARTERY: GPU_OBJECT_VEIN;//code golf!
    float L = state.params.voxel_edge_length;
    int chain_length = 32;

    float r = state.params.pial_artery_radius;

    //roots uniformly distributed (this might run into issues later with axon growth, oh well)
    int grid_width = state.params.grid_width;
    int ix = vessel_id % grid_width;
    int iy = vessel_id / grid_width;

    float spacing = L / static_cast<float>(grid_width + 1);

    float x0 = spacing * static_cast<float>(ix + 1);
    float y0 = spacing * static_cast<float>(iy + 1);

    float z_start = (object_type == GPU_OBJECT_ARTERY) ? state.params.z_top * L : state.params.z_bottom * L;// Arteries go down (start at top), veins oppo
    float dz = (object_type == GPU_OBJECT_ARTERY) ? -1.0f : 1.0f;
    float step = fmaxf(r / state.params.overlap_factor, 0.25f);
    int previous_sphere_id = -1;

    for (int k = 0; k < chain_length; ++k) {
        int sphere_id = start_sphere_id + vessel_id * chain_length + k;
        if (sphere_id >= state.spheres.capacity) {
            *state.error_code = 10;//i really should stop making random error codes
            return;
        }

        float z = z_start + dz * step * static_cast<float>(k);

        if (z < 0.0f || z > L) {//out of bounds...
            return;
        }
        // normally distributed change of direction
        unsigned int seed = static_cast<unsigned int>(state.params.seed + 1234ULL * static_cast<unsigned long long>(sphere_id));
        float rand_x = state.params.bv_norm_dist * r * (random_float(seed + 1u) - state.params.bv_norm_norm);
        float rand_y = state.params.bv_norm_dist * r * (random_float(seed + 2u) - state.params.bv_norm_norm);
        write_sphere(
            state,
            sphere_id,
            x0 + rand_x,
            y0 + rand_y,
            z,
            r,
            object_type,
            vessel_id,
            0,
            previous_sphere_id
        );
        previous_sphere_id = sphere_id;
    }
}
}// end namespace

// void generate_blood_vessels_kernel(GpuSimulationState state, int step) {
// //pseudocode (from: https://pmc.ncbi.nlm.nih.gov/articles/PMC8504684/table/T7/)
// // 1.	FOR i = 1 TO nTerminals DO
// // 2.	 terminal aT = venTerminalSampleGeneratorList.getSample //arterial terminal
// // 3.	 segments vS = choose closest two ∈ close_segment_list(venous_tree, constraintList)
// place in same spatial grid?
//then clear spatial grid
// // 4.	 add_fork (venous_tree, aT, vS[0]) // with or without optimizing
// // 5.	 add_fork (venous_tree, aT, vS[1]) // with or without optimizing
// // 6.	 remove(venTerminalSampleGeneratorList, aT)
// // 7.	 balanceTree(venous_tree, d0) // update venous_tree diameter ratios
// murrays law (r^3[big]=r^3+r^3[small])
// // 8.	 terminal vT = artTerminalSampleGeneratorList.getSample //venous terminal

// // 9.	 segment aS = choose closest two ∈ close_segment_list(arterial_tree, constraintList)
// // 10.	 add_fork (arterial_tree, vT, aS[0]) // with or without optimizing
// // 11.	 add_fork (arterial_tree, vT, aS[1]) // with or without optimizing
// // 12.	 remove(artTerminalSampleGeneratorList, vT)
// // 13.	 balanceTree(artery_tree, d0) // update arterial_tree diameter ratios

// // 14.	ENDFOR
// }
void generate_blood_vessels_gpu(GpuSimulationState& state){
    int sphere_count =0;
    CUDA_CHECK(cudaMemcpy(
        &sphere_count,
        state.spheres.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    int num_blood_vessels = state.params.num_pial_arteries;
    int chain_length = 32;
    int new_spheres = num_blood_vessels*chain_length;
    
    int blocks = gpu_num_blocks(num_blood_vessels);
    generate_blood_vessels_kernel<<<blocks, GPU_THREADS_PER_BLOCK>>>(state, sphere_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int updated_sphere_count = sphere_count + new_spheres;

    CUDA_CHECK(cudaMemcpy(
        state.spheres.count,
        &updated_sphere_count,
        sizeof(int),
        cudaMemcpyHostToDevice
    ));
}
