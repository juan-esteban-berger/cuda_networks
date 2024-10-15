/**
 * @file test_matrix_relu_derivative.cu
 * @brief Unit tests for the Matrix::relu_derivative method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixReluDerivativeTest
 * @brief Test fixture for the Matrix::relu_derivative method tests.
 */
class MatrixReluDerivativeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::relu_derivative correctly applies the ReLU derivative function.
 *
 * This test creates a matrix with positive and negative values,
 * applies the ReLU derivative function, and confirms that all negative values
 * are set to zero and all positive values are set to one.
 */
TEST_F(MatrixReluDerivativeTest, ApplyReluDerivativeCorrectly) {
    // Create a 3x3 matrix with known values
    Matrix m(3, 3);
    double h_data[9] = {-1.0, 0.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0};
    cudaMemcpy(m.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Apply ReLU derivative to the matrix
    Matrix result = m.relu_derivative();

    // Print the original and result matrices
    std::cout << "Original matrix:" << std::endl;
    m.print(2);
    std::cout << "After ReLU derivative:" << std::endl;
    result.print(2);

    // Allocate host memory to verify the results
    double* h_result = new double[9];
    cudaMemcpy(h_result, result.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define the expected result after ReLU derivative
    double expected[9] = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};

    // Check that all elements are correctly processed by ReLU derivative
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(h_result[i], expected[i]);
    }

    // Free the allocated host memory
    delete[] h_result;
}
