/**
 * @file matrix_read_csv_limited.cu
 * @brief Implementation of the Matrix::read_csv_limited method.
 */
#include "matrix.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

void Matrix::read_csv_limited(const char* filename,
                              int startRow,
                              int endRow,
                              int fileRows,
                              int fileCols) {
    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file");
    }

    // Check if the specified range is valid
    if (startRow < 0 || endRow > fileRows || startRow >= endRow) {
        throw std::runtime_error("Invalid row range specified");
    }

    // Check if the matrix dimensions match the specified range and file columns
    if (rows != endRow - startRow || cols != fileCols) {
        throw std::runtime_error("Matrix dimensions do not match the specified range and file columns");
    }

    // Vector to temporarily store the data read from the CSV
    std::vector<double> temp_data(rows * cols);
    std::string line, value;
    int currentRow = 0;

    // Read the CSV file line by line
    while (std::getline(file, line) && currentRow < fileRows) {
        // Process only the rows within the specified range
        if (currentRow >= startRow && currentRow < endRow) {
            std::istringstream s(line);
            for (int col = 0; col < fileCols; ++col) {
                // Parse each value in the line, separated by commas
                if (!std::getline(s, value, ',')) {
                    throw std::runtime_error("Insufficient columns in CSV file");
                }
                // Convert the string value to double and store it in the temporary vector
                temp_data[(currentRow - startRow) * cols + col] = std::stod(value);
            }
        }
        currentRow++;
    }

    // Check if we read enough rows
    if (currentRow < endRow) {
        throw std::runtime_error("Insufficient rows in CSV file");
    }

    // Copy data from the temporary vector to the device (GPU) memory
    cudaMemcpy(d_data, temp_data.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
}
