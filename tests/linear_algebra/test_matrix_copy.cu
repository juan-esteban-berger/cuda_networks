/**
 * @file test_matrix_copy.cu
 * @brief Unit tests for the Matrix::copy method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixCopyTest
 * @brief Test fixture for the Matrix::copy method tests.
 */
class MatrixCopyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::copy creates an identical but separate matrix.
 *
 * This test creates a matrix, makes a copy, and confirms that the copy
 * has the same content but is a distinct object in memory.
 */
TEST_F(MatrixCopyTest, CopyMatrixCorrectly) {
    // Create a 3x3 matrix with known values
    Matrix original(3, 3);
    double h_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    cudaMemcpy(original.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Create a copy of the matrix
    Matrix copy = original.copy();

    // Print the original and copied matrices
    std::cout << "Original matrix:" << std::endl;
    original.print(2);
    std::cout << "Copied matrix:" << std::endl;
    copy.print(2);

    // Verify that the dimensions are the same
    EXPECT_EQ(original.get_rows(), copy.get_rows());
    EXPECT_EQ(original.get_cols(), copy.get_cols());

    // Allocate host memory to verify the results
    double* h_original = new double[9];
    double* h_copy = new double[9];
    cudaMemcpy(h_original, original.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_copy, copy.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are the same
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(h_original[i], h_copy[i]);
    }

    // Verify that the memory addresses are different
    EXPECT_NE(original.get_data(), copy.get_data());

    // Modify the original matrix
    h_data[0] = 10.0;
    cudaMemcpy(original.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Verify that the copy remains unchanged
    cudaMemcpy(h_original, original.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_copy, copy.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_NE(h_original[0], h_copy[0]);

    // Print the matrices after modification
    std::cout << "Original matrix after modification:" << std::endl;
    original.print(2);
    std::cout << "Copied matrix after original's modification:" << std::endl;
    copy.print(2);

    // Free the allocated host memory
    delete[] h_original;
    delete[] h_copy;
}
