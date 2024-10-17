/**
 * @file test_matrix_sigmoid_derivative.cu
 * @brief Unit tests for the Matrix::sigmoid_derivative method.
 */
#include <gtest/gtest.h>
#include "../../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

/**
 * @class MatrixSigmoidDerivativeTest
 * @brief Test fixture for the Matrix::sigmoid_derivative method tests.
 */
class MatrixSigmoidDerivativeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::sigmoid_derivative correctly applies the sigmoid derivative function.
 *
 * This test creates a matrix with various values,
 * applies the sigmoid derivative function, and confirms that the output
 * matches calculated values and is within the expected range.
 */
TEST_F(MatrixSigmoidDerivativeTest, ApplySigmoidDerivativeCorrectly) {
    // Create a 3x3 matrix with known values
    Matrix m(3, 3);
    double h_data[9] = {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
    cudaMemcpy(m.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Apply sigmoid derivative to the matrix
    Matrix result = m.sigmoid_derivative();

    // Print the original and result matrices
    std::cout << "Original matrix:" << std::endl;
    m.print(2);
    std::cout << "After sigmoid derivative:" << std::endl;
    result.print(4);

    // Allocate host memory to verify the results
    double* h_result = new double[9];
    cudaMemcpy(h_result, result.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are correctly processed by sigmoid derivative
    for (int i = 0; i < 9; ++i) {
        // Calculate expected sigmoid value
        double sigmoid_x = 1.0 / (1.0 + std::exp(-h_data[i]));
        
        // Calculate expected sigmoid derivative value
        double expected = sigmoid_x * (1.0 - sigmoid_x);
        
        // Check if the result is within a small epsilon of the expected value
        EXPECT_NEAR(h_result[i], expected, 1e-6);
        
        // Check if the result is within the valid range for sigmoid derivative (0 to 0.25)
        EXPECT_GE(h_result[i], 0.0);
        EXPECT_LE(h_result[i], 0.25);
    }

    // Free the allocated host memory
    delete[] h_result;
}
