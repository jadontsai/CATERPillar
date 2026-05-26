#include "gpu_commit.h"

#include "gpu_runner.h"
#include "gpu_simulation_state.h"
#include "gpu_collision_check.h"
#include "gpu_init.h"
#include "gpu_candidate_generation.h"
#include "gpu_candidate_selection.h"
#include "gpu_in_box.h"
#include "gpu_launch_config.h"
#include "gpu_spatial_grid.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <functional>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err));    \
        }                                                                  \
    } while (0)

__global__//launched from cpu, runs on gpu
//this is just to see if we can actually do anything with the GPU, not used in actual axon stuff
void gpu_smoke_test_kernel(int* error_code) {
    //threadIdx.x is the thread index within the block, blockIdx.x is the block index within the grid
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *error_code = 1234;
        //if it prints this, then this worked
    }
}


float time_cuda_stage_ms(
    const std::function<void()>& stage
) {//timing gpu stages
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    stage();
    //done
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    //print
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}
//this is designed as writing temp candidates (probably could collapse this and the 
// other csv function if we were crunched for space); not actually used for real results
//useful for debugging though
void write_csv(const std::string& filename,
GpuSimulationState& state,
int num_to_write){//num_to_write_is some limit if you needed it, make it arbitrarily large if you want
    int total_candidates = state.candidates.total_candidates;

    num_to_write = std::min(num_to_write, total_candidates);
    std::vector<float> h_x(num_to_write);
    std::vector<float> h_y(num_to_write);   
    std::vector<float> h_z(num_to_write);
    std::vector<float> h_r(num_to_write);
    std::vector<float> dir_x(num_to_write); 
    std::vector<float> dir_y(num_to_write);  
    std::vector<float> dir_z(num_to_write);
    std::vector<int> front_id(num_to_write);
    std::vector<int> valid(num_to_write);
    std::vector<int> selected(num_to_write);

    CUDA_CHECK(cudaMemcpy(h_x.data(), state.candidates.x, num_to_write * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), state.candidates.y, num_to_write * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z.data(), state.candidates.z, num_to_write * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_r.data(), state.candidates.r, num_to_write * sizeof(float), cudaMemcpyDeviceToHost));   
    CUDA_CHECK(cudaMemcpy(dir_x.data(), state.candidates.dir_x, num_to_write * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dir_y.data(), state.candidates.dir_y, num_to_write * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dir_z.data(), state.candidates.dir_z, num_to_write * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(front_id.data(), state.candidates.front_id, num_to_write * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(valid.data(), state.candidates.valid, num_to_write * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(selected.data(), state.candidates.selected, num_to_write * sizeof(int), cudaMemcpyDeviceToHost)); 

    std::ofstream file(filename);

    if(!file){
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    file << "x,y,z,r,dir_x,dir_y,dir_z,front_id,valid,selected\n";
    for (int i = 0; i < num_to_write; ++i) {
        //it might want swc, just change that here
        file << h_x[i] << "," << h_y[i] << "," << h_z[i] << "," << h_r[i] << "," << dir_x[i] << "," << dir_y[i] << "," << dir_z[i] << "," << front_id[i] << "," << valid[i] << "," << selected[i] << "\n";
    }
    std::cout << "Wrote " << num_to_write << " candidates to " << filename << std::endl;
}

void write_final_csv(
    const std::string& filename,
    GpuSimulationState& state
) {//only writes commited spheres
    int sphere_count = 0;

    CUDA_CHECK(cudaMemcpy(
        &sphere_count,
        state.spheres.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    std::vector<float> h_x(sphere_count);
    std::vector<float> h_y(sphere_count);
    std::vector<float> h_z(sphere_count);
    std::vector<float> h_r(sphere_count);
    std::vector<int> h_object_type(sphere_count);
    std::vector<int> h_object_id(sphere_count);
    std::vector<int> h_branch_id(sphere_count);
    std::vector<int> h_parent_sphere_id(sphere_count);

    CUDA_CHECK(cudaMemcpy(h_x.data(), state.spheres.x, sphere_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), state.spheres.y, sphere_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z.data(), state.spheres.z, sphere_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_r.data(), state.spheres.r, sphere_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_object_type.data(), state.spheres.object_type, sphere_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_object_id.data(), state.spheres.object_id, sphere_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_branch_id.data(), state.spheres.branch_id, sphere_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_parent_sphere_id.data(), state.spheres.parent_sphere_id, sphere_count * sizeof(int), cudaMemcpyDeviceToHost));

    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error("Could not write to" + filename);
    }
    file << "sphere_id,x,y,z,r,object_type,object_id,branch_id,parent_sphere_id\n";

    for (int i = 0; i < sphere_count; ++i) {
        file << i << ","
             << h_x[i] << ","
             << h_y[i] << ","
             << h_z[i] << ","
             << h_r[i] << ","
             << h_object_type[i] << ","
             << h_object_id[i] << ","
             << h_branch_id[i] << ","
             << h_parent_sphere_id[i] << "\n";
    }
    std::cout << "Wrote " << sphere_count
              << " spheres to " << filename << std::endl;
}
void run_gpu_simulation(const GpuParameters& params) {
    std::cout << "doing gpu stuff..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    GpuSimulationState state;
    allocate_gpu_state(state, params);
    GpuSpatialGrid grid;
    allocate_gpu_spatial_grid(
        grid,
        2.0f * params.min_radius,
        params.voxel_edge_length,
        params.max_spheres,
        params.max_spheres * 16
    );
    //so state is on the cpu stack, but it's pointing to GPU memory

    initialize_scene_gpu(state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "after initislize_scene" << std::endl;

    for (int i = 0; i <1000; ++i){
    float grid_ms = time_cuda_stage_ms([&]() {
        build_gpu_spatial_grid(state, grid);
    });

    float gen_ms = time_cuda_stage_ms([&]() {
        gpu_generate_candidates(state, step);
    });

    float inbox_ms = time_cuda_stage_ms([&]() {
        run_in_box_check(state);
    });

    float collision_ms = time_cuda_stage_ms([&]() {
        run_collision_check(state, grid);
    });

    float select_ms = time_cuda_stage_ms([&]() {
        select_valid_candidate_gpu(state);
    });

    float commit_ms = time_cuda_stage_ms([&]() {
        commit_candidates_gpu(state);
    });

    std::cout << "step " << step
          << " grid_ms=" << grid_ms
          << " gen_ms=" << gen_ms
          << " inbox_ms=" << inbox_ms
          << " collision_ms=" << collision_ms
          << " select_ms=" << select_ms
          << " commit_ms=" << commit_ms
          << std::endl;
    //     build_gpu_spatial_grid(state, grid);
    //     std::cout << "after build_gpu_spatial_grid" << i << std::endl;

    //     gpu_generate_candidates(state, i);
    //     std::cout << "after gpu_generate_candidates" << std::endl;

    //     run_in_box_check(state);
    //     std::cout << "after run_in_box_check" << std::endl;
    //     run_collision_check(state, grid);
    //     std::cout << "after run_collision_check" << std::endl;

    //     select_valid_candidate_gpu(state);
    //     std::cout << "after select_valid_candidate_gpu" << std::endl;
    //     //write_csv("candidates_step_" + std::to_string(i) + ".csv", state, 258);
    //    // std::cout << "after write_csv_temp" << std::endl;
    //     commit_candidates_gpu(state);
    //     std::cout << "after commit_candidates_gpu" << std::endl;
    }

    //"spheres" is the input if you wanna see the commited stuff
    write_final_csv("final.csv", state);
    std::cout << "after write_csv_final" << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "GPU simulation time (ms): " << elapsed_ms << std::endl;

    int sphere_count = 0;

    CUDA_CHECK(cudaMemcpy(
        &sphere_count,
        state.spheres.count,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    std::cout << "sphere count after commit: "
            << sphere_count << std::endl;
    free_gpu_spatial_grid(grid);   
    std::cout << "freed spatial grid" << std::endl;

    free_gpu_state(state);
    std::cout << "freed gpu state" << std::endl;


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
    //launches the single front kernel with 1 block and 1 thread

    //cpu variables that will be copied back from gpu
    // int sphere_count =0;
    // int front_count = 0;

    // float x_0 = 0.0f;
    // float y_0 = 0.0f;
    // float z_0 = 0.0f;
    // float r_0 = 0.0f;

    //actually copying things back
    // CUDA_CHECK(cudaMemcpy(
    //     &sphere_count,
    //     state.spheres.count,
    //     sizeof(int),
    //     cudaMemcpyDeviceToHost
    // ));
    // //expected value of 1, since there's 1 sphere
    // CUDA_CHECK(cudaMemcpy(
    //     &front_count,
    //     state.fronts.count,
    //     sizeof(int),
    //     cudaMemcpyDeviceToHost
    // ));
    // //expected value of 1, since there's 1 front
    // CUDA_CHECK(cudaMemcpy(
    //     &x_0,
    //     state.spheres.x,
    //     sizeof(float),
    //     cudaMemcpyDeviceToHost
    // ));
    // CUDA_CHECK(cudaMemcpy(
    //     &y_0,
    //     state.spheres.y,
    //     sizeof(float),
    //     cudaMemcpyDeviceToHost
    // ));
    // CUDA_CHECK(cudaMemcpy(
    //     &z_0,
    //     state.spheres.z,
    //     sizeof(float),
    //     cudaMemcpyDeviceToHost
    // ));
    // CUDA_CHECK(cudaMemcpy(
    //     &r_0,
    //     state.spheres.r,
    //     sizeof(float),
    //     cudaMemcpyDeviceToHost
    // ));

    // std::cout << "sphere count (should be 1): " << sphere_count << std::endl;
    // std::cout << "front count (should be 1): " << front_count << std::endl;
    // std::cout << "sphere 0 position (should be half voxel edge length): " << x_0 << ", " << y_0 << ", " << z_0 << std::endl;
    // std::cout << "sphere 0 radius (should be min radius): " << r_0 << std::endl;
}