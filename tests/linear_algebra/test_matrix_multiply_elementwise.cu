/**
 * @file test_matrix_multiply_elementwise.cu
 * @brief Unit tests for the Matrix::multiply_elementwise method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixMultiplyElementwiseTest
 * @brief Test fixture for the Matrix::multiply_elementwise method tests.
 */
class MatrixMultiplyElementwiseTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::multiply_elementwise correctly multiplies two matrices element-wise.
 */
TEST_F(MatrixMultiplyElementwiseTest, MultiplyMatricesElementwiseCorrectly) {
    // Create test matrices
    Matrix m1(2, 3);
    Matrix m2(2, 3);

    // Prepare test data
    double h_m1[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double h_m2[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    // Copy test data to GPU
    cudaMemcpy(m1.get_data(), h_m1, 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m2.get_data(), h_m2, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Perform element-wise matrix multiplication
    Matrix result = m1.multiply_elementwise(m2);

    // Print matrices for visual verification
    std::cout << "Matrix 1:" << std::endl;
    m1.print(2);
    std::cout << "Matrix 2:" << std::endl;
    m2.print(2);
    std::cout << "Result of element-wise multiplication:" << std::endl;
    result.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[6];
    cudaMemcpy(h_result, result.get_data(), 6 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define expected result
    double expected[6] = {7.0, 16.0, 27.0, 40.0, 55.0, 72.0};

    // Verify result
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(h_result[i], expected[i]);
    }

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Matrix::multiply_elementwise throws an exception for incompatible dimensions.
 */
TEST_F(MatrixMultiplyElementwiseTest, ThrowsExceptionForIncompatibleDimensions) {
    // Create matrices with incompatible dimensions
    Matrix m1(2, 3);
    Matrix m2(3, 2);

    // Verify that an exception is thrown
    EXPECT_THROW(m1.multiply_elementwise(m2), std::invalid_argument);
}
