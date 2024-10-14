/**
 * @file matrix_print.cu
 * @brief Implementation of the Matrix::print method.
 */
#include "matrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

void Matrix::print() {
    // Allocate host memory to copy the data from GPU
    double* h_data = new double[rows * cols];
    
    // Copy data from GPU (device) to CPU (host)
    cudaMemcpy(h_data, d_data, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Print matrix dimensions
    std::cout << "Matrix " << rows << "x" << cols << ":" << std::endl;
    
    // Iterate through the matrix and print each element
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Format each number with fixed precision and alignment
            std::cout << std::setw(9) << std::fixed << std::setprecision(3) << std::setfill(' ') 
                      << (h_data[i * cols + j] >= 0 ? " " : "") // Extra space for positive numbers
                      << h_data[i * cols + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Free the allocated host memory
    delete[] h_data;
}
