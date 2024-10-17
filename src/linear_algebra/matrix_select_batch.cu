/**
 * @file matrix_select_batch.cu
 * @brief Implementation of the Matrix::select_batch method for selecting a subset of the matrix.
 */

#include "matrix.h"
#include <cuda_runtime.h>
#include <stdexcept>

/**
 * @brief CUDA kernel for selecting a subset of the matrix.
 * @param src Pointer to the source matrix data.
 * @param dst Pointer to the destination matrix data.
 * @param src_cols Number of columns in the source matrix.
 * @param dst_cols Number of columns in the destination matrix.
 * @param start_row Starting row index.
 * @param start_col Starting column index.
 * @param num_rows Number of rows to select.
 * @param num_cols Number of columns to select.
 */
__global__ void selectBatchKernel(const double* src, double* dst, int src_cols, int dst_cols, 
                                  int start_row, int start_col, int num_rows, int num_cols) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within the selected subset bounds
    if (row < num_rows && col < num_cols) {
        // Calculate source and destination indices
        int src_idx = (start_row + row) * src_cols + (start_col + col);
        int dst_idx = row * dst_cols + col;

        // Copy the element from source to destination
        dst[dst_idx] = src[src_idx];
    }
}

Matrix Matrix::select_batch(int start_row, int end_row, int start_col, int end_col) const {
    // Validate input ranges
    if (start_row < 0 || end_row > rows || start_col < 0 || end_col > cols ||
        start_row >= end_row || start_col >= end_col) {
        throw std::out_of_range("Invalid row or column range specified");
    }

    // Calculate dimensions of the selected subset
    int num_rows = end_row - start_row;
    int num_cols = end_col - start_col;

    // Create a new matrix to store the selected subset
    Matrix result(num_rows, num_cols);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA kernel
    selectBatchKernel<<<numBlocks, threadsPerBlock>>>(
        d_data, result.d_data, cols, num_cols, start_row, start_col, num_rows, num_cols
    );

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();

    return result;
}
