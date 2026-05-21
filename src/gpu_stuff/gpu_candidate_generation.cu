#include "gpu_candidate_generation.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cmath>
//for the hashing stuff

#define CUDA_CHECK(call)                                                   \
      do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err));    \
        }                                                                  \
      } while (0)

__device__//called from gpu, runs on gpu
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

__global__
void kernel_generate_candidates(GpuSimulationState state, int step) {
      //kernel that actually generates candidates
      int front_id = blockIdx.x;
      //for now we're doing one block for one front, so if there's 32 threads we get 32 candindates per front
      int attempt_id = threadIdx.x;

      int front_count = *state.fronts.count;
      int candidates_per_front = state.candidates.candidates_per_front;
      //should be 32

      if (front_id >= front_count || attempt_id >= candidates_per_front ||state.fronts.active[front_id] == 0) {
            //any of the stopping conditions (does not exist, inactive, out of bounds)
            return;
      }
      int candidate_id = front_id * candidates_per_front + attempt_id;
      //some unique number (0 indexed btw)
      unsigned int seed = static_cast<unsigned int>(//casts because the origin function takes 32 bits
            state.params.seed
            + 1234567ULL * static_cast<unsigned long long>(step)
            + 8967ULL * static_cast<unsigned long long>(front_id)
            + 67ULL * static_cast<unsigned long long>(attempt_id)
      );

      //some random direction
      float u1 = random_float(seed+1);
      float u2 = random_float(seed+2);

      //some angle within 2 pi
      float theta = u1 * 2.0f * 3.14159265f;

      //some point on the unit sphere
      float z = u2 * 2.0f - 1.0f;
      //radius of vector (since unit sphere has vector x^2 + y^2 + z^2 = 1) at that z value is sqrt(1-z^2)
      float xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
      //directions
      float dx = xy * cosf(theta);
      float dy = xy * sinf(theta);
      float dz = z;

      //CHANGE THIS LATER WHEN I IMPLEMENT GAMMA DISTRIBUTION
      //==========================================================================
      float radius = state.params.min_radius;
      float overlap_factor = fmaxf(1.0f, state.params.overlap_factor);
      //error handling if the factor is less than 1 for whatever reason, but maybe that should be more graceful

      float space = fmax(radius, state.fronts.r[front_id]) / overlap_factor;

      
      //store candidate
      state.candidates.x[candidate_id] = state.fronts.x[front_id] + dx * space;
      state.candidates.y[candidate_id] = state.fronts.y[front_id] + dy * space;
      state.candidates.z[candidate_id] = state.fronts.z[front_id] + dz * space;
      state.candidates.r[candidate_id] = radius;

      //store direction
      state.candidates.dir_x[candidate_id] = dx;
      state.candidates.dir_y[candidate_id] = dy;
      state.candidates.dir_z[candidate_id] = dz;

      //metadata
      state.candidates.front_id[candidate_id] = front_id;
      state.candidates.valid[candidate_id] = 1;
      //valid for now, didnt actually check anything
      state.candidates.selected[candidate_id] = 0;
      return;
}

void gpu_generate_candidates(GpuSimulationState& state, int step) {
      int front_count = 0;
      CUDA_CHECK(cudaMemcpy(
            &front_count,
            state.fronts.count,
            sizeof(int),
            cudaMemcpyDeviceToHost
      ));

      if (front_count <= 0) {
            //something went wrong here
            return;
      }
      int threads_per_block = state.candidates.candidates_per_front;
      int blocks = front_count;

      kernel_generate_candidates<<<blocks, threads_per_block>>>(state, step);
      //probably 32 threads per block
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

}


