/**
 * @file matrix_argmax.cu
 * @brief GPU implementation of the column-wise argmax function for matrices.
 */

#include "matrix.h"
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for computing the argmax of each column in a matrix.
 * @param m Pointer to the matrix data on the GPU.
 * @param result Pointer to the result vector on the GPU.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
__global__ void argmax_GPU(const double *m, double *result, int rows, int cols) {
    // Determine the column this thread is responsible for
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Proceed if the column index is within matrix bounds
    if (col < cols) {
        // Initialize max_val with the first element in the column and max_idx to the first row
        double max_val = m[col];
        int max_idx = 0;

        // Iterate through the rows to find the maximum value in the column
        for (int row = 1; row < rows; row++) {
            double val = m[row * cols + col]; // Access element (row, col)
            if (val > max_val) {
                max_val = val;
                max_idx = row;
            }
        }
        
        // Store the index of the maximum value in the result vector for this column
        result[col] = static_cast<double>(max_idx);
    }
}

/**
 * @brief Launches the argmax_GPU kernel to perform column-wise argmax on the matrix.
 * @return A Vector containing the row indices of the maximum values for each column.
 */
Vector Matrix::argmax() const {
    // Create a result vector on the device
    Vector result(cols);

    // Define grid and block sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the argmax kernel on the device
    argmax_GPU<<<blocksPerGrid, threadsPerBlock>>>(d_data, result.get_data(), rows, cols);

    // Ensure the kernel execution is complete
    cudaDeviceSynchronize();

    return result;
}
