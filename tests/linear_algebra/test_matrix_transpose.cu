/**
 * @file test_matrix_transpose.cu
 * @brief Unit tests for the Matrix::transpose method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixTransposeTest
 * @brief Test fixture for the Matrix::transpose method tests.
 */
class MatrixTransposeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::transpose correctly transposes a 2x3 matrix.
 */
TEST_F(MatrixTransposeTest, Transpose2x3Matrix) {
    // Create a 2x3 matrix with known values
    Matrix m(2, 3);
    double h_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    cudaMemcpy(m.get_data(), h_data, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Transpose the matrix
    Matrix transposed = m.transpose();

    // Print the original and transposed matrices
    std::cout << "Original matrix (2x3):" << std::endl;
    m.print(2);
    std::cout << "Transposed matrix (3x2):" << std::endl;
    transposed.print(2);

    // Verify the transposed result on the CPU
    double expected[6] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    double* h_transposed = new double[6];
    cudaMemcpy(h_transposed, transposed.get_data(), 6 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that each element is correctly transposed
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(h_transposed[i], expected[i]);
    }

    // Clean up
    delete[] h_transposed;
}

/**
 * @test
 * @brief Verify that Matrix::transpose correctly transposes a square matrix (3x3).
 */
TEST_F(MatrixTransposeTest, Transpose3x3Matrix) {
    // Create a 3x3 matrix with known values
    Matrix m(3, 3);
    double h_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    cudaMemcpy(m.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Transpose the matrix
    Matrix transposed = m.transpose();

    // Print the original and transposed matrices
    std::cout << "Original matrix (3x3):" << std::endl;
    m.print(2);
    std::cout << "Transposed matrix (3x3):" << std::endl;
    transposed.print(2);

    // Verify the transposed result on the CPU
    double expected[9] = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
    double* h_transposed = new double[9];
    cudaMemcpy(h_transposed, transposed.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that each element is correctly transposed
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(h_transposed[i], expected[i]);
    }

    // Clean up
    delete[] h_transposed;
}
