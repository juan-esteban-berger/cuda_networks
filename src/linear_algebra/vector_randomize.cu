/**
 * @file vector_randomize.cu
 * @brief Implementation of the Vector::randomize method to fill the vector with random values.
 */

#include "vector.h"
#include <curand_kernel.h>  // Include CUDA random library for random number generation
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel function that fills each element in the vector with a random value between -0.5 and 0.5.
 * @param data Pointer to the vector data on the GPU.
 * @param rows Number of elements in the vector.
 * @param seed Seed for random number generator.
 */
__global__ void randomizeKernel(double* data, int rows, unsigned long seed) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread's index is within the vector bounds
    if (idx < rows) {
        // Initialize a random state for this thread
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate a random value between 0 and 1, then shift it to be between -0.5 and 0.5
        double randomValue = curand_uniform(&state) - 0.5;

        // Assign the random value to the current vector element
        data[idx] = randomValue;
    }
}

/**
 * @brief Fills the vector with random values between -0.5 and 0.5.
 */
void Vector::randomize() {
    // Define the number of threads per block (a common choice for good occupancy)
    int threadsPerBlock = 256;

    // Calculate the number of blocks needed to cover all elements
    // We use ceiling division to ensure we have enough blocks
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    // Generate a seed for the random number generator
    // We use the current time to ensure different seeds across runs
    unsigned long seed = time(NULL);

    // Launch the CUDA kernel
    randomizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, rows, seed);

    // Wait for the kernel to complete before returning
    // This ensures all random values are generated before any subsequent operations
    cudaDeviceSynchronize();
}
