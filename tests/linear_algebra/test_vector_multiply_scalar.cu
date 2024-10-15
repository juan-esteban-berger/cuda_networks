/**
 * @file test_vector_multiply_scalar.cu
 * @brief Unit tests for the Vector::multiply_scalar method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>

/**
 * @class VectorMultiplyScalarTest
 * @brief Test fixture for the Vector::multiply_scalar method tests.
 */
class VectorMultiplyScalarTest : public ::testing::Test {
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
 * @brief Verify that Vector::multiply_scalar correctly multiplies all elements by a scalar.
 */
TEST_F(VectorMultiplyScalarTest, MultiplyVectorByScalarCorrectly) {
    // Create test vector
    Vector v(5);

    // Prepare test data
    double h_v[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Copy test data to GPU
    cudaMemcpy(v.get_data(), h_v, 5 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original vector
    std::cout << "Original vector:" << std::endl;
    v.print(2);

    // Perform multiplication by scalar
    double scalar = 2.5;
    v.multiply_scalar(scalar);

    // Print result
    std::cout << "Vector after multiplying by " << scalar << ":" << std::endl;
    v.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[5];
    cudaMemcpy(h_result, v.get_data(), 5 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define expected result
    double expected[5] = {2.5, 5.0, 7.5, 10.0, 12.5};

    // Verify result
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(h_result[i], expected[i]);
    }

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Vector::multiply_scalar handles multiplication by a very small number correctly.
 */
TEST_F(VectorMultiplyScalarTest, MultiplyByVerySmallNumber) {
    // Create test vector
    Vector v(4);
    
    // Prepare test data
    double h_v[4] = {1.0, -1.0, 0.0, 2.0};
    cudaMemcpy(v.get_data(), h_v, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original vector
    std::cout << "Original vector:" << std::endl;
    v.print(2);

    // Perform multiplication by a very small number
    double scalar = DBL_EPSILON;
    std::cout << "Multiplying by scalar: " << scalar << std::endl;
    v.multiply_scalar(scalar);

    // Print result
    std::cout << "Vector after multiplying by very small number:" << std::endl;
    v.print(10);  // Using higher precision to show small values

    // Copy result back to CPU for verification
    double* h_result = new double[4];
    cudaMemcpy(h_result, v.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify results
    EXPECT_DOUBLE_EQ(h_result[0], DBL_EPSILON);
    EXPECT_DOUBLE_EQ(h_result[1], -DBL_EPSILON);
    EXPECT_DOUBLE_EQ(h_result[2], 0.0);
    EXPECT_DOUBLE_EQ(h_result[3], 2 * DBL_EPSILON);

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Vector::multiply_scalar handles multiplication of very large numbers correctly.
 */
TEST_F(VectorMultiplyScalarTest, MultiplyLargeNumbers) {
    // Create test vector
    Vector v(4);
    
    // Prepare test data with very large numbers
    double h_v[4] = {DBL_MAX / 2, -DBL_MAX / 2, DBL_MAX / 4, -DBL_MAX / 4};
    cudaMemcpy(v.get_data(), h_v, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original vector
    std::cout << "Original vector with large numbers:" << std::endl;
    v.print(4);

    // Perform multiplication
    double scalar = 2.0;
    std::cout << "Multiplying by scalar: " << scalar << std::endl;
    v.multiply_scalar(scalar);

    // Print result
    std::cout << "Vector after multiplying large numbers:" << std::endl;
    v.print(4);

    // Copy result back to CPU for verification
    double* h_result = new double[4];
    cudaMemcpy(h_result, v.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify results using approximate equality
    EXPECT_TRUE(isApproximatelyEqual(h_result[0], DBL_MAX, 1e-6));
    EXPECT_TRUE(isApproximatelyEqual(h_result[1], -DBL_MAX, 1e-6));
    EXPECT_TRUE(isApproximatelyEqual(h_result[2], DBL_MAX / 2, 1e-6));
    EXPECT_TRUE(isApproximatelyEqual(h_result[3], -DBL_MAX / 2, 1e-6));

    // Clean up
    delete[] h_result;
}
