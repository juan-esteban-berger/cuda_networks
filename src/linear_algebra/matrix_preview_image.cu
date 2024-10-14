/**
 * @file matrix_preview_image.cu
 * @brief Implementation of the Matrix::preview_image method.
 */
#include "matrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>

void Matrix::preview_image(int row_index, int image_size_x, int image_size_y) const {
    // Check if the row_index is within the valid range
    if (row_index < 0 || row_index >= rows) {
        throw std::out_of_range("Invalid row index");
    }

    // Check if the image dimensions fit within the matrix columns
    if (image_size_x * image_size_y > cols) {
        throw std::invalid_argument("Image dimensions exceed matrix column count");
    }

    // Allocate host memory to store a single row of the matrix
    double* h_data = new double[cols];

    // Copy the specified row from device (GPU) memory to host memory
    cudaMemcpy(h_data, d_data + row_index * cols, cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Iterate over each row of the image
    for (int i = 0; i < image_size_x; ++i) {
        // Iterate over each column of the image
        for (int j = 0; j < image_size_y; ++j) {
            // Calculate the index in the flattened array
            int index = i * image_size_y + j;

            // Round the pixel value to the nearest integer
            int value = static_cast<int>(std::round(h_data[index]));

            // Print spaces for zero values (background)
            if (value == 0) {
                std::cout << "    ";
            } else {
                // Print non-zero values with 3-digit width
                std::cout << std::setw(3) << value << " ";
            }
        }
        // Move to the next line after each row of the image
        std::cout << std::endl;
    }
    // Print an extra newline for separation
    std::cout << std::endl;

    // Free the allocated host memory
    delete[] h_data;
}
