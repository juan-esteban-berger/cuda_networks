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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;

    // Ensure the thread's index is within the matrix bounds
    if (idx < totalElements) {
        int row = idx / cols;
        int col = idx % cols;

        // Initialize a random state for this thread
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Assign a random value between -0.5 and 0.5 to the current matrix element
        data[row * cols + col] = curand_uniform(&state) - 0.5;
    }
}

/**
 * @brief Fills the matrix with random values between -0.5 and 0.5.
 */
void Matrix::randomize() {
    int totalElements = rows * cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long seed = time(NULL);

    randomizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, rows, cols, seed);
    cudaDeviceSynchronize();
}
