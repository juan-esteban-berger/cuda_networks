/**
 * @file test_matrix_select_batch.cu
 * @brief Test file for the Matrix::select_batch method.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "../../src/linear_algebra/matrix.h"

TEST(MatrixTest, SelectBatch) {
    // Create a 5x5 matrix with known values
    Matrix m(5, 5);
    double* h_data = new double[25];
    for (int i = 0; i < 25; ++i) {
        h_data[i] = i;
    }
    cudaMemcpy(m.get_data(), h_data, 25 * sizeof(double), cudaMemcpyHostToDevice);

    // Print the original matrix
    std::cout << "Original Matrix:" << std::endl;
    m.print(0);

    // Select a 3x3 subset from the middle of the matrix
    Matrix subset = m.select_batch(1, 4, 1, 4);

    // Print the selected subset
    std::cout << "Selected Subset:" << std::endl;
    subset.print(0);

    // Verify the dimensions of the subset
    EXPECT_EQ(subset.get_rows(), 3);
    EXPECT_EQ(subset.get_cols(), 3);

    // Verify the contents of the subset
    double* h_subset = new double[9];
    cudaMemcpy(h_subset, subset.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);

    double expected_subset[9] = {6, 7, 8, 11, 12, 13, 16, 17, 18};
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(h_subset[i], expected_subset[i]);
        if (h_subset[i] != expected_subset[i]) {
            std::cout << "Mismatch at index " << i << ": Expected " << expected_subset[i] 
                      << ", Got " << h_subset[i] << std::endl;
        }
    }

    // Test error handling for invalid ranges
    EXPECT_THROW(m.select_batch(-1, 3, 0, 3), std::out_of_range);
    EXPECT_THROW(m.select_batch(0, 6, 0, 3), std::out_of_range);
    EXPECT_THROW(m.select_batch(0, 3, -1, 3), std::out_of_range);
    EXPECT_THROW(m.select_batch(0, 3, 0, 6), std::out_of_range);
    EXPECT_THROW(m.select_batch(3, 2, 0, 3), std::out_of_range);
    EXPECT_THROW(m.select_batch(0, 3, 3, 2), std::out_of_range);

    // Clean up
    delete[] h_data;
    delete[] h_subset;
}
