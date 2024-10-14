/**
 * @file matrix_randomize.cu
 * @brief Implementation of the Matrix::randomize method to fill the matrix with random values.
 */

#include "matrix.h"
#include <curand_kernel.h>  // Include CUDA random library for random number generation
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel function that fills each element in the matrix with a random value between -0.5 and 0.5.
 * @param data Pointer to the matrix data on the GPU.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param seed Seed for random number generator.
 */
__global__ void randomizeKernel(double* data, int rows, int cols, unsigned long seed) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;

    // Ensure the thread's index is within the matrix bounds
    if (idx < totalElements) {
        // Calculate the row and column for this thread
        int row = idx / cols;
        int col = idx % cols;

        // Initialize a random state for this thread
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate a random value between 0 and 1, then shift it to be between -0.5 and 0.5
        double randomValue = curand_uniform(&state) - 0.5;

        // Assign the random value to the current matrix element
        data[row * cols + col] = randomValue;
    }
}

/**
 * @brief Fills the matrix with random values between -0.5 and 0.5.
 */
void Matrix::randomize() {
    // Calculate the total number of elements in the matrix
    int totalElements = rows * cols;

    // Define the number of threads per block (a common choice for good occupancy)
    int threadsPerBlock = 256;

    // Calculate the number of blocks needed to cover all elements
    // We use ceiling division to ensure we have enough blocks
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Generate a seed for the random number generator
    // We use the current time to ensure different seeds across runs
    unsigned long seed = time(NULL);

    // Launch the CUDA kernel
    randomizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, rows, cols, seed);

    // Wait for the kernel to complete before returning
    // This ensures all random values are generated before any subsequent operations
    cudaDeviceSynchronize();
}
