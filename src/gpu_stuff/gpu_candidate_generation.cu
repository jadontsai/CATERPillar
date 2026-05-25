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
__device__
float safe_rsqrt(float x) {
      return rsqrtf(fmaxf(x, 1e-12f));
      //a bit safer behaviour than 0
}

__device__
void normalize3(float& x, float& y, float& z) {
      float inv_norm = safe_rsqrt(x * x + y * y + z * z);
      x *= inv_norm;
      y *= inv_norm;
      z *= inv_norm;
}

__device__
void make_persistent_direction(
      float old_dx,
      float old_dy,
      float old_dz,
      float rand_dx,
      float rand_dy,
      float rand_dz,
      float persistence,
      float& out_dx,
      float& out_dy,
      float& out_dz
) {
      normalize3(old_dx, old_dy, old_dz);
      normalize3(rand_dx, rand_dy, rand_dz);

      float randomness = 1.0f - persistence;
      //higher persistence means straighter

      out_dx = persistence * old_dx + randomness * rand_dx;
      out_dy = persistence * old_dy + randomness * rand_dy;
      out_dz = persistence * old_dz + randomness * rand_dz;

      normalize3(out_dx, out_dy, out_dz);

      // Prevent backward turns
      float dot_prev =
            out_dx * old_dx +
            out_dy * old_dy +
            out_dz * old_dz;

      if (dot_prev < 0.0f) {
            out_dx = old_dx;
            out_dy = old_dy;
            out_dz = old_dz;
      }
}

__device__
float sample_gamma_integer(unsigned int seed, int shape, float scale){
      float sum = 0.0f;
      for(int i = 0; i< shape; ++i){
            float u = random_float(seed + 1234u * static_cast<unsigned int>(i));
            u = fmaxf(u, 1e-7f);
            sum += -logf(u);
      }
      return scale *sum;
      }

__device__
float sample_standard_normal_box_muller(unsigned int seed) {
      //normally distributed radii for axon 
      // pertubation (like how much each sphere changes)
      float u1 = random_float(seed + 1u);
      float u2 = random_float(seed + 2u);

      u1 = fmaxf(u1, 1e-7f);

      float mag = sqrtf(-2.0f * logf(u1));
      float angle = 2.0f * 3.141592f * u2;

      return mag * cosf(angle);
}

__device__
float sample_normal_box_muller(
      unsigned int seed,
      float mean,
      float stddev
) {
      return mean + stddev * sample_standard_normal_box_muller(seed);
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

      //BETTER RANDOM BEHAVIOUR
      float rand_dx = xy * cosf(theta);
      float rand_dy = xy * sinf(theta);
      float rand_dz = z;

      // Current front direction
      float old_dx = state.fronts.dir_x[front_id];
      float old_dy = state.fronts.dir_y[front_id];
      float old_dz = state.fronts.dir_z[front_id];
      float persistence = 0.90f;
      float dx;
      float dy;
      float dz;

      make_persistent_direction(
            old_dx,
            old_dy,
            old_dz,
            rand_dx,
            rand_dy,
            rand_dz,
            persistence,
            dx,
            dy,
            dz
      );
      //CHANGE THIS LATER WHEN I IMPLEMENT GAMMA DISTRIBUTION
      //==========================================================================
      //float radius = state.params.min_radius;
      int shape = static_cast<int>(roundf(state.params.alpha));
      //float radius = sample_gamma_integer(seed + 50000u, shape, state.params.beta);
      float previous_radius = state.fronts.r[front_id];
      //Temporary beading strength.
      float radius_stddev = 0.05f * previous_radius;
      float radius = sample_normal_box_muller(seed + 50000u,previous_radius, radius_stddev);

      radius = fmaxf(radius, state.params.min_radius);
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
      if (!state.fronts.parent_sphere_id) {
            //if parent sphere id is null, just set it to -1
            state.candidates.parent_id[candidate_id] = -1;
      } else {
            state.candidates.parent_id[candidate_id] = state.fronts.parent_sphere_id[front_id];
      }
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


