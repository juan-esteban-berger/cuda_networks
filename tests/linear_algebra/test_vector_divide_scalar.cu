/**
 * @file test_vector_divide_scalar.cu
 * @brief Unit tests for the Vector::divide_scalar method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>

/**
 * @class VectorDivideScalarTest
 * @brief Test fixture for the Vector::divide_scalar method tests.
 */
class VectorDivideScalarTest : public ::testing::Test {
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
 * @brief Verify that Vector::divide_scalar correctly divides all elements by a scalar.
 */
TEST_F(VectorDivideScalarTest, DivideVectorByScalarCorrectly) {
    // Create test vector
    Vector v(5);

    // Prepare test data
    double h_v[5] = {2.0, 4.0, 6.0, 8.0, 10.0};

    // Copy test data to GPU
    cudaMemcpy(v.get_data(), h_v, 5 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original vector
    std::cout << "Original vector:" << std::endl;
    v.print(2);

    // Perform division by scalar
    double scalar = 2.0;
    v.divide_scalar(scalar);

    // Print result
    std::cout << "Vector after dividing by " << scalar << ":" << std::endl;
    v.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[5];
    cudaMemcpy(h_result, v.get_data(), 5 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define expected result
    double expected[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Verify result
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(h_result[i], expected[i]);
    }

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Vector::divide_scalar handles division by a very small number correctly.
 */
TEST_F(VectorDivideScalarTest, DivideByVerySmallNumber) {
    // Create test vector
    Vector v(4);
    
    // Prepare test data
    double h_v[4] = {1.0, -1.0, 0.0, 2.0};
    cudaMemcpy(v.get_data(), h_v, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original vector
    std::cout << "Original vector:" << std::endl;
    v.print(2);

    // Perform division by a very small number
    double scalar = DBL_EPSILON / 2.0;  // Very small number
    std::cout << "Dividing by scalar: " << scalar << std::endl;
    v.divide_scalar(scalar);

    // Print result
    std::cout << "Vector after dividing by very small number:" << std::endl;
    v.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[4];
    cudaMemcpy(h_result, v.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

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
 * @brief Verify that Vector::divide_scalar handles division of very large numbers correctly.
 */
TEST_F(VectorDivideScalarTest, DivideLargeNumbers) {
    // Create test vector
    Vector v(4);
    
    // Prepare test data with very large numbers
    double h_v[4] = {DBL_MAX, -DBL_MAX, DBL_MAX / 2, -DBL_MAX / 2};
    cudaMemcpy(v.get_data(), h_v, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original vector
    std::cout << "Original vector with large numbers:" << std::endl;
    v.print(4);

    // Perform division
    double scalar = 2.0;
    std::cout << "Dividing by scalar: " << scalar << std::endl;
    v.divide_scalar(scalar);

    // Print result
    std::cout << "Vector after dividing large numbers:" << std::endl;
    v.print(4);

    // Copy result back to CPU for verification
    double* h_result = new double[4];
    cudaMemcpy(h_result, v.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

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
 * @brief Verify that Vector::divide_scalar throws an exception when dividing by exactly zero.
 */
TEST_F(VectorDivideScalarTest, ThrowsExceptionWhenDividingByExactlyZero) {
    // Create test vector
    Vector v(4);
    
    // Attempt to divide by zero and expect an exception
    EXPECT_THROW(v.divide_scalar(0.0), std::invalid_argument);
    
    std::cout << "Successfully caught exception when dividing by zero." << std::endl;
}
