/**
 * @file vector_multiply_scalar.cu
 * @brief Implementation of the Vector::multiply_scalar method for GPU-accelerated multiplication of a vector by a scalar.
 */

#include "vector.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cfloat>
#include <cmath>

/**
 * @brief CUDA kernel for multiplying vector elements by a scalar.
 * @param data Pointer to the vector data.
 * @param scalar The scalar to multiply by.
 * @param size Total number of elements in the vector.
 */
__global__ void vectorMultiplyScalarKernel(double* data, double scalar, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within vector bounds
    if (idx < size) {
        // Perform multiplication
        double result = data[idx] * scalar;

        // Handle overflow
        if (!isfinite(result)) {
            result = (result > 0.0) ? DBL_MAX : -DBL_MAX;
        }

        // Store the result
        data[idx] = result;
    }
}

/**
 * @brief Multiplies all elements in the vector by a scalar.
 * @param scalar The scalar to multiply by.
 */
void Vector::multiply_scalar(double scalar) {
    // Calculate total number of elements
    int size = rows;

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    vectorMultiplyScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, size);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();
}
