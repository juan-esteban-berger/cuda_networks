/**
 * @file test_matrix_sum.cu
 * @brief Unit tests for the Matrix::sum method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixSumTest
 * @brief Test fixture for the Matrix::sum method tests.
 */
class MatrixSumTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::sum correctly sums all elements in a matrix.
 */
TEST_F(MatrixSumTest, SumMatrixElementsCorrectly) {
    // Create test matrix
    Matrix m(2, 3);

    // Prepare test data
    double h_m[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Copy test data to GPU
    cudaMemcpy(m.get_data(), h_m, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Print matrix for visual verification
    std::cout << "Matrix:" << std::endl;
    m.print(2);

    // Perform matrix sum
    double result = m.sum();

    std::cout << "Sum of all elements: " << result << std::endl;

    // Define expected result
    double expected = 21.0;  // 1 + 2 + 3 + 4 + 5 + 6

    // Verify result
    EXPECT_DOUBLE_EQ(result, expected);
}

/**
 * @test
 * @brief Verify that Matrix::sum correctly handles a matrix with negative and positive values.
 */
TEST_F(MatrixSumTest, SumMatrixWithNegativeValues) {
    // Create test matrix
    Matrix m(2, 2);

    // Prepare test data with negative and positive values
    double h_m[4] = {-1.0, 2.0, -3.0, 4.0};

    // Copy test data to GPU
    cudaMemcpy(m.get_data(), h_m, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Print matrix for visual verification
    std::cout << "Matrix with negative values:" << std::endl;
    m.print(2);

    // Perform matrix sum
    double result = m.sum();

    std::cout << "Sum of all elements: " << result << std::endl;

    // Define expected result
    double expected = 2.0;  // -1 + 2 + -3 + 4

    // Verify result
    EXPECT_DOUBLE_EQ(result, expected);
}
