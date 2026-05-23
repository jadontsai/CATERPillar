#include "gpu_simulation_state.h"

#include <cuda_runtime.h>//for the usual cuda functions

#include <stdexcept>
#include <string>

//helper function to do malloc (returns errors better)
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err));    \
        }                                                                  \
    } while (0)

static void allocate_spheres(GpuSphereTable& spheres, int capacity) {
    //making the arrays
    spheres.capacity = capacity;
    //9 variables
    CUDA_CHECK(cudaMalloc(&spheres.x, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&spheres.y, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&spheres.z, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&spheres.r, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&spheres.object_type, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&spheres.object_id, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&spheres.branch_id, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&spheres.parent_sphere_id, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&spheres.count, sizeof(int)));

    //memset initializes stuff, in this case the count is zero at the start
    CUDA_CHECK(cudaMemset(spheres.count, 0, sizeof(int)));
}

static void allocate_fronts(GpuGrowthFrontTable& fronts, int capacity) {

//making more arrays
    fronts.capacity = capacity;
    //13 variables
    CUDA_CHECK(cudaMalloc(&fronts.x, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fronts.y, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fronts.z, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fronts.r, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fronts.dir_x, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fronts.dir_y, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fronts.dir_z, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fronts.object_type, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&fronts.object_id, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&fronts.branch_id, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&fronts.parent_sphere_id, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&fronts.active, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&fronts.count, sizeof(int)));

    //setting variables
    CUDA_CHECK(cudaMemset(fronts.active, 0, capacity * sizeof(int)));
    CUDA_CHECK(cudaMemset(fronts.count, 0, sizeof(int)));
}

static void allocate_candidates(
    GpuCandidateTable& candidates,
    int max_fronts,
    int candidates_per_front){
    candidates.candidates_per_front = candidates_per_front;
    candidates.total_candidates = max_fronts * candidates_per_front;

    const int n = candidates.total_candidates;
    //10 variables
    CUDA_CHECK(cudaMalloc(&candidates.x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&candidates.y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&candidates.z, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&candidates.r, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&candidates.dir_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&candidates.dir_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&candidates.dir_z, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&candidates.front_id, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&candidates.valid, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&candidates.selected, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&candidates.parent_id, n * sizeof(int)));

    //setting variables to zero
    CUDA_CHECK(cudaMemset(candidates.valid, 0, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(candidates.selected, 0, n * sizeof(int)));
}

void allocate_gpu_state(
    GpuSimulationState& state, 
    const GpuParameters& params) 
    {
    //the actual function we wanna call, so not static
    state.params = params;

    allocate_spheres(state.spheres, params.max_spheres);
    allocate_fronts(state.fronts, params.max_growth_fronts);
    allocate_candidates(
        state.candidates,
        params.max_growth_fronts,
        params.candidates_per_front
    );

    //2 variables (done and error flags)
    CUDA_CHECK(cudaMalloc(&state.done, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.error_code, sizeof(int)));

    CUDA_CHECK(cudaMemset(state.done, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(state.error_code, 0, sizeof(int)));
}

void free_gpu_state(GpuSimulationState& state) {
    //equivalnt of dealloc
    cudaFree(state.spheres.x);
    state.spheres.x=nullptr;
    cudaFree(state.spheres.y);
    state.spheres.y=nullptr;
    cudaFree(state.spheres.z);
    state.spheres.z=nullptr;
    cudaFree(state.spheres.r);
    state.spheres.r=nullptr;
    cudaFree(state.spheres.object_type);
    state.spheres.object_type=nullptr;
    cudaFree(state.spheres.object_id);
    state.spheres.object_id=nullptr;
    cudaFree(state.spheres.branch_id);
    state.spheres.branch_id=nullptr;
    cudaFree(state.spheres.parent_sphere_id);
    state.spheres.parent_sphere_id=nullptr;
    cudaFree(state.spheres.count);
    state.spheres.count=nullptr;
    cudaFree(state.fronts.x);
    state.fronts.x=nullptr;
    cudaFree(state.fronts.y);
    state.fronts.y=nullptr;
    cudaFree(state.fronts.z);
    state.fronts.z=nullptr;
    cudaFree(state.fronts.r);
    state.fronts.r=nullptr;
    cudaFree(state.fronts.dir_x);
    state.fronts.dir_x=nullptr;
    cudaFree(state.fronts.dir_y);
    state.fronts.dir_y=nullptr;
    cudaFree(state.fronts.dir_z);
    state.fronts.dir_z=nullptr;
    cudaFree(state.fronts.object_type);
    state.fronts.object_type=nullptr;
    cudaFree(state.fronts.object_id);
    state.fronts.object_id=nullptr;
    cudaFree(state.fronts.branch_id);
    state.fronts.branch_id=nullptr;
    cudaFree(state.fronts.parent_sphere_id);
    state.fronts.parent_sphere_id=nullptr;                  
    cudaFree(state.fronts.active);
    state.fronts.active=nullptr;
    cudaFree(state.fronts.count);
    state.fronts.count=nullptr;
    cudaFree(state.candidates.x);
    state.candidates.x=nullptr;
    cudaFree(state.candidates.y);
    state.candidates.y=nullptr;
    cudaFree(state.candidates.z);
    state.candidates.z=nullptr;
    cudaFree(state.candidates.r);
    state.candidates.r=nullptr;
    cudaFree(state.candidates.dir_x);
    state.candidates.dir_x=nullptr;
    cudaFree(state.candidates.dir_y);
    state.candidates.dir_y=nullptr;
    cudaFree(state.candidates.dir_z);
    state.candidates.dir_z=nullptr;
    cudaFree(state.candidates.front_id);
    state.candidates.front_id=nullptr;
    cudaFree(state.candidates.valid);
    state.candidates.valid=nullptr;
    cudaFree(state.candidates.parent_id);
    state.candidates.parent_id = nullptr;
    cudaFree(state.candidates.selected);
    state.candidates.selected=nullptr;
    cudaFree(state.done);
    state.done=nullptr;
    cudaFree(state.error_code);
    state.error_code=nullptr;
    //sanity check 139-106=34 variables freed,
    // 9 in spheres, 13 in fronts, 10 in candidates, 
    // 2 in state
    //woah i can add 
}