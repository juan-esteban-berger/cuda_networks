/**
 * @file test_vector_copy.cu
 * @brief Unit tests for the Vector::copy method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class VectorCopyTest
 * @brief Test fixture for the Vector::copy method tests.
 */
class VectorCopyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Vector::copy creates an identical but separate vector.
 *
 * This test creates a vector, makes a copy, and confirms that the copy
 * has the same content but is a distinct object in memory.
 */
TEST_F(VectorCopyTest, CopyVectorCorrectly) {
    // Create a vector with known values
    Vector original(5);
    double h_data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    cudaMemcpy(original.get_data(), h_data, 5 * sizeof(double), cudaMemcpyHostToDevice);

    // Create a copy of the vector
    Vector copy = original.copy();

    // Print the original and copied vectors
    std::cout << "Original vector:" << std::endl;
    original.print(2);
    std::cout << "Copied vector:" << std::endl;
    copy.print(2);

    // Verify that the dimensions are the same
    EXPECT_EQ(original.get_rows(), copy.get_rows());

    // Allocate host memory to verify the results
    double* h_original = new double[5];
    double* h_copy = new double[5];
    cudaMemcpy(h_original, original.get_data(), 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_copy, copy.get_data(), 5 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are the same
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(h_original[i], h_copy[i]);
    }

    // Verify that the memory addresses are different
    EXPECT_NE(original.get_data(), copy.get_data());

    // Modify the original vector
    h_data[0] = 10.0;
    cudaMemcpy(original.get_data(), h_data, 5 * sizeof(double), cudaMemcpyHostToDevice);

    // Verify that the copy remains unchanged
    cudaMemcpy(h_original, original.get_data(), 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_copy, copy.get_data(), 5 * sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_NE(h_original[0], h_copy[0]);

    // Print the vectors after modification
    std::cout << "Original vector after modification:" << std::endl;
    original.print(2);
    std::cout << "Copied vector after original's modification:" << std::endl;
    copy.print(2);

    // Free the allocated host memory
    delete[] h_original;
    delete[] h_copy;
}
