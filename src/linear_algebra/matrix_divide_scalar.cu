/**
 * @file matrix_divide_scalar.cu
 * @brief Implementation of the Matrix::divide_scalar method for GPU-accelerated division of a matrix by a scalar.
 */

#include "matrix.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cfloat>
#include <cmath>

/**
 * @brief CUDA kernel for dividing matrix elements by a scalar.
 * @param data Pointer to the matrix data.
 * @param scalar The scalar to divide by.
 * @param size Total number of elements in the matrix.
 */
__global__ void divideScalarKernel(double* data, double scalar, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix bounds
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
 * @brief Divides all elements in the matrix by a scalar.
 * @param scalar The scalar to divide by.
 * @throws std::invalid_argument if scalar is exactly zero.
 */
void Matrix::divide_scalar(double scalar) {
    // Check for division by exactly zero
    if (scalar == 0.0) {
        throw std::invalid_argument("Cannot divide by exactly zero");
    }

    // Calculate total number of elements
    int size = rows * cols;

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    divideScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, size);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();
}
