/**
 * @file test_matrix_divide_scalar.cu
 * @brief Unit tests for the Matrix::divide_scalar method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>

/**
 * @class MatrixDivideScalarTest
 * @brief Test fixture for the Matrix::divide_scalar method tests.
 */
class MatrixDivideScalarTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper function to check if two doubles are approximately equal
    bool isApproximatelyEqual(double a, double b, double epsilon = 1e-6) {
        if (std::isinf(a) && std::isinf(b)) {
            return (a > 0) == (b > 0);
        }
        if (std::abs(a) > DBL_MAX / 2 || std::abs(b) > DBL_MAX / 2) {
            // For very large numbers, use a relative error
            return std::abs(a - b) / std::max(std::abs(a), std::abs(b)) < epsilon;
        }
        return std::abs(a - b) < epsilon;
    }
};

/**
 * @test
 * @brief Verify that Matrix::divide_scalar correctly divides all elements by a scalar.
 */
TEST_F(MatrixDivideScalarTest, DivideMatrixByScalarCorrectly) {
    // Create test matrix
    Matrix m(2, 3);

    // Prepare test data
    double h_m[6] = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};

    // Copy test data to GPU
    cudaMemcpy(m.get_data(), h_m, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original matrix
    std::cout << "Original matrix:" << std::endl;
    m.print(2);

    // Perform division by scalar
    double scalar = 2.0;
    m.divide_scalar(scalar);

    // Print result
    std::cout << "Matrix after dividing by " << scalar << ":" << std::endl;
    m.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[6];
    cudaMemcpy(h_result, m.get_data(), 6 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define expected result
    double expected[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Verify result
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(h_result[i], expected[i]);
    }

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Matrix::divide_scalar handles division by a very small number correctly.
 */
TEST_F(MatrixDivideScalarTest, DivideByVerySmallNumber) {
    // Create test matrix
    Matrix m(2, 2);
    
    // Prepare test data
    double h_m[4] = {1.0, -1.0, 0.0, 2.0};
    cudaMemcpy(m.get_data(), h_m, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original matrix
    std::cout << "Original matrix:" << std::endl;
    m.print(2);

    // Perform division by a very small number
    double scalar = DBL_EPSILON / 2.0;  // Very small number
    std::cout << "Dividing by scalar: " << scalar << std::endl;
    m.divide_scalar(scalar);

    // Print result
    std::cout << "Matrix after dividing by very small number:" << std::endl;
    m.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[4];
    cudaMemcpy(h_result, m.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify results
    EXPECT_DOUBLE_EQ(h_result[0], DBL_MAX);
    EXPECT_DOUBLE_EQ(h_result[1], -DBL_MAX);
    EXPECT_DOUBLE_EQ(h_result[2], 0.0);
    EXPECT_DOUBLE_EQ(h_result[3], DBL_MAX);

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Matrix::divide_scalar handles division of very large numbers correctly.
 */
TEST_F(MatrixDivideScalarTest, DivideLargeNumbers) {
    // Create test matrix
    Matrix m(2, 2);
    
    // Prepare test data with very large numbers
    double h_m[4] = {DBL_MAX, -DBL_MAX, DBL_MAX / 2, -DBL_MAX / 2};
    cudaMemcpy(m.get_data(), h_m, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original matrix
    std::cout << "Original matrix with large numbers:" << std::endl;
    m.print(4);

    // Perform division
    double scalar = 2.0;
    std::cout << "Dividing by scalar: " << scalar << std::endl;
    m.divide_scalar(scalar);

    // Print result
    std::cout << "Matrix after dividing large numbers:" << std::endl;
    m.print(4);

    // Copy result back to CPU for verification
    double* h_result = new double[4];
    cudaMemcpy(h_result, m.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify results using approximate equality
    EXPECT_TRUE(isApproximatelyEqual(h_result[0], DBL_MAX, 1e-6));
    EXPECT_TRUE(isApproximatelyEqual(h_result[1], -DBL_MAX, 1e-6));
    EXPECT_TRUE(isApproximatelyEqual(h_result[2], DBL_MAX / 4, 1e-6));
    EXPECT_TRUE(isApproximatelyEqual(h_result[3], -DBL_MAX / 4, 1e-6));

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Matrix::divide_scalar throws an exception when dividing by exactly zero.
 */
TEST_F(MatrixDivideScalarTest, ThrowsExceptionWhenDividingByExactlyZero) {
    // Create test matrix
    Matrix m(2, 2);
    
    // Attempt to divide by zero and expect an exception
    EXPECT_THROW(m.divide_scalar(0.0), std::invalid_argument);
    
    std::cout << "Successfully caught exception when dividing by zero." << std::endl;
}
