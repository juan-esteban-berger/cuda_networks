/**
 * @file test_vector_initialize.cu
 * @brief Unit tests for the Vector::initialize method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class VectorInitializeTest
 * @brief Test fixture for the Vector::initialize method tests.
 */
class VectorInitializeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Vector::initialize sets all elements to zero.
 *
 * This test creates a vector, initializes it, and confirms that all
 * elements are set to zero. It also prints the vector for visual verification.
 */
TEST_F(VectorInitializeTest, InitializeToZero) {
    Vector v(10);  // Create a 10-element vector
    v.initialize();  // Initialize the vector (should set all elements to 0)

    // Print the vector to verify initialization
    std::cout << "Printing initialized 10-element vector:\n";
    v.print(2);

    // Allocate host memory to verify the results
    double* h_data = new double[10];
    cudaMemcpy(h_data, v.get_data(), 10 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are indeed 0
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(h_data[i], 0.0);
    }

    // Free the allocated host memory
    delete[] h_data;
}
