/**
 * @file test_vector_subtract_scalar.cu
 * @brief Unit tests for the Vector::subtract_scalar method.
 */

#include <gtest/gtest.h>
#include "../../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

/**
 * @class VectorSubtractScalarTest
 * @brief Test fixture for the Vector::subtract_scalar method tests.
 */
class VectorSubtractScalarTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Helper function to check if two doubles are approximately equal
     * @param a First value
     * @param b Second value
     * @param epsilon Tolerance for comparison
     * @return true if values are approximately equal, false otherwise
     */
    bool isApproximatelyEqual(double a, double b, double epsilon = 1e-6) {
        return std::fabs(a - b) < epsilon;
    }
};

/**
 * @test
 * @brief Verify that Vector::subtract_scalar correctly subtracts a scalar from all elements.
 */
TEST_F(VectorSubtractScalarTest, SubtractScalarTest) {
    // Create a vector
    Vector v(5);

    // Initialize vector with known values
    double h_data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    cudaMemcpy(v.get_data(), h_data, 5 * sizeof(double), cudaMemcpyHostToDevice);

    // Subtract scalar
    double scalar = 1.5;
    v.subtract_scalar(scalar);

    // Copy result back to host
    double h_result[5];
    cudaMemcpy(h_result, v.get_data(), 5 * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(isApproximatelyEqual(h_result[i], h_data[i] - scalar));
    }

    // Print results
    std::cout << "Vector after subtracting " << scalar << ":" << std::endl;
    v.print(4);
}

/**
 * @test
 * @brief Verify that Vector::subtract_scalar handles edge cases correctly.
 */
TEST_F(VectorSubtractScalarTest, SubtractScalarEdgeCases) {
    // Create a vector
    Vector v(3);

    // Test case 1: Subtracting from zero
    double h_data1[3] = {0.0, 0.0, 0.0};
    cudaMemcpy(v.get_data(), h_data1, 3 * sizeof(double), cudaMemcpyHostToDevice);
    v.subtract_scalar(1.0);
    double h_result1[3];
    cudaMemcpy(h_result1, v.get_data(), 3 * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(isApproximatelyEqual(h_result1[i], -1.0));
    }

    // Test case 2: Subtracting a very large number
    double h_data2[3] = {1.0, 2.0, 3.0};
    cudaMemcpy(v.get_data(), h_data2, 3 * sizeof(double), cudaMemcpyHostToDevice);
    v.subtract_scalar(DBL_MAX / 2);
    double h_result2[3];
    cudaMemcpy(h_result2, v.get_data(), 3 * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; ++i) {
        EXPECT_LT(h_result2[i], 0.0);
    }

    // Print results
    std::cout << "Vector after edge case tests:" << std::endl;
    v.print(4);
}
