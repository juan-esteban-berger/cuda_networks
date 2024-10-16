/**
 * @file matrix_copy.cu
 * @brief Implementation of copy and move operations for the Matrix class.
 */

#include "matrix.h"
#include <cuda_runtime.h>

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
    // Allocate new memory on the device
    cudaMalloc(&d_data, rows * cols * sizeof(double));
    // Copy data from the other matrix to this one
    cudaMemcpy(d_data, other.d_data, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {  // Protect against self-assignment
        // Free existing memory
        cudaFree(d_data);
        
        // Copy dimensions
        rows = other.rows;
        cols = other.cols;
        
        // Allocate new memory
        cudaMalloc(&d_data, rows * cols * sizeof(double));
        // Copy data from the other matrix
        cudaMemcpy(d_data, other.d_data, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows(other.rows), cols(other.cols), d_data(other.d_data) {
    // Transfer ownership and reset the source object
    other.d_data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {  // Protect against self-assignment
        // Free existing memory
        cudaFree(d_data);
        
        // Transfer ownership
        rows = other.rows;
        cols = other.cols;
        d_data = other.d_data;
        
        // Reset the source object
        other.d_data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

Matrix Matrix::copy() const {
    // Use the copy constructor to create a deep copy
    return *this;
}
