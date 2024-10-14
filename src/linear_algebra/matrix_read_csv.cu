/**
 * @file matrix_read_csv.cu
 * @brief Implementation of the Matrix::read_csv method.
 */
#include "matrix.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

void Matrix::read_csv(const char* filename) {
    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file");
    }

    // Vector to temporarily store the data read from the CSV
    std::vector<double> temp_data;
    std::string line, value;

    // Read the CSV file line by line
    while (std::getline(file, line)) {
        // Create a string stream from the current line
        std::istringstream s(line);
        
        // Parse each value in the line, separated by commas
        while (std::getline(s, value, ',')) {
            // Convert the string value to double and add it to the temporary vector
            temp_data.push_back(std::stod(value));
        }
    }

    // Check if the number of values read matches the matrix dimensions
    if (temp_data.size() != rows * cols) {
        throw std::runtime_error("CSV data size does not match matrix dimensions");
    }

    // Copy data from the temporary vector to the device (GPU) memory
    cudaMemcpy(d_data, temp_data.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
}
