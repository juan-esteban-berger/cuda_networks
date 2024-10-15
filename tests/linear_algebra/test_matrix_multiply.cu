/**
 * @file test_matrix_multiply.cu
 * @brief Unit tests for the Matrix::multiply method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixMultiplyTest
 * @brief Test fixture for the Matrix::multiply method tests.
 */
class MatrixMultiplyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::multiply correctly multiplies two matrices.
 */
TEST_F(MatrixMultiplyTest, MultiplyMatricesCorrectly) {
    // Create test matrices
    Matrix m1(2, 3);
    Matrix m2(3, 2);

    // Prepare test data
    double h_m1[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double h_m2[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    // Copy test data to GPU
    cudaMemcpy(m1.get_data(), h_m1, 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m2.get_data(), h_m2, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Perform matrix multiplication
    Matrix result = m1.multiply(m2);

    // Print matrices for visual verification
    std::cout << "Matrix 1:" << std::endl;
    m1.print(2);
    std::cout << "Matrix 2:" << std::endl;
    m2.print(2);
    std::cout << "Result of multiplication:" << std::endl;
    result.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[4];
    cudaMemcpy(h_result, result.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define expected result
    double expected[4] = {58.0, 64.0, 139.0, 154.0};

    // Verify result
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(h_result[i], expected[i]);
    }

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Matrix::multiply throws an exception for incompatible dimensions.
 */
TEST_F(MatrixMultiplyTest, ThrowsExceptionForIncompatibleDimensions) {
    // Create matrices with incompatible dimensions
    Matrix m1(2, 3);
    Matrix m2(2, 2);

    // Verify that an exception is thrown
    EXPECT_THROW(m1.multiply(m2), std::invalid_argument);
}
