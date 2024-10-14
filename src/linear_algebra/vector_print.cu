/**
 * @file vector_print.cu
 * @brief Implementation of the Vector::print method with consistent spacing.
 */
#include "vector.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdio>

void Vector::print(int decimals) {
    // Create format string for desired number of decimals
    char format[20];
    sprintf(format, "%%d:\t%%.%df\n", decimals);

    // Allocate host memory to copy the data from GPU
    double* h_data = new double[rows];
    cudaMemcpy(h_data, d_data, rows * sizeof(double), cudaMemcpyDeviceToHost);

    // Print vector dimensions
    std::cout << "Vector with " << rows << " rows:\n";

    // Print column header (since vector is treated as a single column)
    std::cout << "\t0:\t\n";

    // Iterate over rows
    for (int i = 0; i < rows; ++i) {
        // If more than 10 rows, only print first and last 5
        if (i == 5 && rows > 10) {
            std::cout << "...\t...\n";
            i = rows - 5;  // Skip to the last 5 rows
        }
        // Print row index and value
        printf(format, i, h_data[i]);
    }

    // Free the allocated host memory
    delete[] h_data;
    std::cout << std::endl;
}
