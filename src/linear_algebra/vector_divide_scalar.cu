/**
 * @file vector_divide_scalar.cu
 * @brief Implementation of the Vector::divide_scalar method for GPU-accelerated division of a vector by a scalar.
 */

#include "vector.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cfloat>
#include <cmath>

/**
 * @brief CUDA kernel for dividing vector elements by a scalar.
 * @param data Pointer to the vector data.
 * @param scalar The scalar to divide by.
 * @param size Total number of elements in the vector.
 */
__global__ void vectorDivideScalarKernel(double* data, double scalar, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within vector bounds
    if (idx < size) {
        // Handle division by very small numbers
        if (fabs(scalar) < DBL_EPSILON) {
            // If data is zero, keep it zero
            // Otherwise, set to max or min based on sign
            data[idx] = (data[idx] == 0.0) ? 0.0 : ((data[idx] > 0.0) ? DBL_MAX : -DBL_MAX);
        } 
        // Handle very large numbers
        else if (fabs(data[idx]) > DBL_MAX / 2) {
            // Preserve sign and set to max value
            data[idx] = (data[idx] > 0.0) ? DBL_MAX : -DBL_MAX;
        } 
        // Regular division for normal cases
        else {
            data[idx] /= scalar;
        }
    }
}

/**
 * @brief Divides all elements in the vector by a scalar.
 * @param scalar The scalar to divide by.
 * @throws std::invalid_argument if scalar is exactly zero.
 */
void Vector::divide_scalar(double scalar) {
    // Check for division by exactly zero
    if (scalar == 0.0) {
        throw std::invalid_argument("Cannot divide by exactly zero");
    }

    // Calculate total number of elements
    int size = rows;

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    vectorDivideScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, size);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();
}
