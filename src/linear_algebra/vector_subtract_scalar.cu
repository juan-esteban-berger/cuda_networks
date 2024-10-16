/**
 * @file vector_subtract_scalar.cu
 * @brief Implementation of the Vector::subtract_scalar method for GPU-accelerated subtraction of a scalar from a vector.
 */

#include "vector.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cfloat>
#include <cmath>

/**
 * @brief CUDA kernel for subtracting a scalar from vector elements.
 * @param data Pointer to the vector data.
 * @param scalar The scalar to subtract.
 * @param size Total number of elements in the vector.
 */
__global__ void vectorSubtractScalarKernel(double* data, double scalar, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within vector bounds
    if (idx < size) {
        // Perform subtraction
        double result = data[idx] - scalar;

        // Handle underflow
        if (!isfinite(result)) {
            result = (result > 0.0) ? DBL_MAX : -DBL_MAX;
        }

        // Store the result
        data[idx] = result;
    }
}

/**
 * @brief Subtracts a scalar value from all elements in the vector.
 * @param scalar The scalar value to subtract.
 */
void Vector::subtract_scalar(double scalar) {
    // Calculate total number of elements
    int size = rows;

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    vectorSubtractScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, size);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();
}
