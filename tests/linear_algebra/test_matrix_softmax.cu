/**
 * @file test_matrix_softmax.cu
 * @brief Unit tests for the Matrix::softmax method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

/**
 * @class MatrixSoftmaxTest
 * @brief Test fixture for the Matrix::softmax method tests.
 */
class MatrixSoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper function to check if two doubles are approximately equal
    bool isApproximatelyEqual(double a, double b, double epsilon = 1e-6) {
        return std::fabs(a - b) < epsilon;
    }

    // Helper function to check if the sum of a column is approximately 1
    bool isSumApproximatelyOne(const double* data, int rows, int cols, int col) {
        double sum = 0.0;
        for (int row = 0; row < rows; ++row) {
            sum += data[row * cols + col];
        }
        return isApproximatelyEqual(sum, 1.0);
    }
};

/**
 * @test
 * @brief Verify that Matrix::softmax correctly applies the softmax function.
 *
 * This test creates a matrix with various values, applies the softmax function,
 * and confirms that the output meets the properties of a softmax distribution.
 */
TEST_F(MatrixSoftmaxTest, ApplySoftmaxCorrectly) {
    // Create a 3x3 matrix with known values
    Matrix m(3, 3);
    double h_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    cudaMemcpy(m.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Apply softmax to the matrix
    Matrix result = m.softmax();

    // Print the original and result matrices
    std::cout << "Original matrix:" << std::endl;
    m.print(4);
    std::cout << "After softmax:" << std::endl;
    result.print(4);

    // Allocate host memory to verify the results
    double* h_result = new double[9];
    cudaMemcpy(h_result, result.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are between 0 and 1
    for (int i = 0; i < 9; ++i) {
        EXPECT_GE(h_result[i], 0.0);
        EXPECT_LE(h_result[i], 1.0);
    }

    // Check that the sum of each column is approximately 1
    for (int col = 0; col < 3; ++col) {
        EXPECT_TRUE(isSumApproximatelyOne(h_result, 3, 3, col));
    }

    // Free the allocated host memory
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Matrix::softmax handles extreme values correctly.
 *
 * This test creates a matrix with very large and very small values to ensure
 * the softmax function handles numerical stability issues correctly.
 */
TEST_F(MatrixSoftmaxTest, HandleExtremeValues) {
    // Create a 2x3 matrix with extreme values
    Matrix m(2, 3);
    double h_data[6] = {1e30, -1e30, 0.0, 1e-30, -1e-30, 1.0};
    cudaMemcpy(m.get_data(), h_data, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Apply softmax to the matrix
    Matrix result = m.softmax();

    // Print the original and result matrices
    std::cout << "Original matrix with extreme values:" << std::endl;
    m.print(4);
    std::cout << "After softmax:" << std::endl;
    result.print(4);

    // Allocate host memory to verify the results
    double* h_result = new double[6];
    cudaMemcpy(h_result, result.get_data(), 6 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are between 0 and 1
    for (int i = 0; i < 6; ++i) {
        EXPECT_GE(h_result[i], 0.0);
        EXPECT_LE(h_result[i], 1.0);
    }

    // Check that the sum of each column is approximately 1
    for (int col = 0; col < 3; ++col) {
        EXPECT_TRUE(isSumApproximatelyOne(h_result, 2, 3, col));
    }

    // Free the allocated host memory
    delete[] h_result;
}
