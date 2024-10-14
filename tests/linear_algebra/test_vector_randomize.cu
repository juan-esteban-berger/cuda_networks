/**
 * @file test_vector_randomize.cu
 * @brief Unit test for the Vector::randomize method to verify random value assignment.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class VectorRandomizeTest
 * @brief Test fixture for testing the Vector::randomize method.
 */
class VectorRandomizeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Vector::randomize assigns values between -0.5 and 0.5 to all elements.
 *
 * This test initializes a vector, applies the randomize method, and confirms that all
 * values are within the specified range. It also prints the vector for manual verification.
 */
TEST_F(VectorRandomizeTest, RandomizeValuesInRange) {
    Vector v(15);  // Create a 15-element vector instance for testing

    v.randomize();  // Apply randomization to the vector elements

    // Print the vector to visually verify the randomized values
    std::cout << "Printing randomized 15-element vector:" << std::endl;
    v.print(3);

    double* h_data = new double[15];  // Allocate host memory to copy vector data from the device

    // Copy the vector data from the device (GPU) to the host (CPU)
    cudaMemcpy(h_data, v.get_data(), 15 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that each element in the vector is within the range [-0.5, 0.5]
    for (int i = 0; i < 15; ++i) {
        EXPECT_GE(h_data[i], -0.5);  // Verify that the value is greater than or equal to -0.5
        EXPECT_LE(h_data[i], 0.5);   // Verify that the value is less than or equal to 0.5
    }

    delete[] h_data;  // Free the allocated host memory
}
