/**
 * @file test_matrix_subtract.cu
 * @brief Unit tests for the Matrix::subtract method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixSubtractTest
 * @brief Test fixture for the Matrix::subtract method tests.
 */
class MatrixSubtractTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::subtract correctly subtracts two matrices.
 */
TEST_F(MatrixSubtractTest, SubtractMatricesCorrectly) {
    // Create test matrices
    Matrix m1(2, 3);
    Matrix m2(2, 3);

    // Prepare test data
    double h_m1[6] = {5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
    double h_m2[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Copy test data to GPU
    cudaMemcpy(m1.get_data(), h_m1, 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m2.get_data(), h_m2, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Perform matrix subtraction
    Matrix result = m1.subtract(m2);

    // Print matrices for visual verification
    std::cout << "Matrix 1:" << std::endl;
    m1.print(2);
    std::cout << "Matrix 2:" << std::endl;
    m2.print(2);
    std::cout << "Result of subtraction:" << std::endl;
    result.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[6];
    cudaMemcpy(h_result, result.get_data(), 6 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define expected result
    double expected[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Verify result
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(h_result[i], expected[i]);
    }

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Matrix::subtract throws an exception for incompatible dimensions.
 */
TEST_F(MatrixSubtractTest, ThrowsExceptionForIncompatibleDimensions) {
    // Create matrices with incompatible dimensions
    Matrix m1(2, 3);
    Matrix m2(3, 2);

    // Verify that an exception is thrown
    EXPECT_THROW(m1.subtract(m2), std::invalid_argument);
}
