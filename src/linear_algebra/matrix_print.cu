/**
 * @file matrix_print.cu
 * @brief Implementation of the Matrix::print method with consistent spacing.
 */
#include "matrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdio>

void Matrix::print(int decimals) {
    // Create format string for desired number of decimals
    char format[20];
    sprintf(format, "%%.%df", decimals);

    // Allocate host memory to copy the data from GPU
    double* h_data = new double[rows * cols];
    cudaMemcpy(h_data, d_data, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Print matrix dimensions
    std::cout << "Matrix with " << rows << " rows and " << cols << " columns:\n";

    // Print column labels
    std::cout << "\t";
    for (int j = 0; j < cols; ++j) {
        if (j == 4 && cols > 8) {
            std::cout << "...\t";
            j = cols - 4;  // Skip to the last 4 columns
        }
        std::cout << j << ":\t";
    }
    std::cout << "\n";

    // Iterate over rows
    for (int i = 0; i < rows; ++i) {
        if (i == 5 && rows > 10) {
            std::cout << "...\n\t";
            for (int k = 0; k < cols; ++k) {
                if (k == 4 && cols > 8) {
                    std::cout << "...\t";
                    k = cols - 4;
                }
                std::cout << "...\t";
            }
            std::cout << "\n";
            i = rows - 5;  // Jump to the last 5 rows
        }

        // Print row index
        std::cout << i << ":\t";

        // Print each element in the row
        for (int j = 0; j < cols; ++j) {
            if (j == 4 && cols > 8) {
                std::cout << "...\t";
                j = cols - 4;  // Skip to the last 4 columns
            }
            printf(format, h_data[i * cols + j]);
            std::cout << "\t";
        }
        std::cout << "\n";
    }

    // Free the allocated host memory
    delete[] h_data;
    std::cout << std::endl;
}
