/**
 * @file test_matrix_randomize.cu
 * @brief Unit test for the Matrix::randomize method to verify random value assignment.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixRandomizeTest
 * @brief Test fixture for testing the Matrix::randomize method.
 */
class MatrixRandomizeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::randomize assigns values between -0.5 and 0.5 to all elements.
 *
 * This test initializes a matrix, applies the randomize method, and confirms that all
 * values are within the specified range. It also prints the matrix for manual verification.
 */
TEST_F(MatrixRandomizeTest, RandomizeValuesInRange) {
    Matrix m(3, 4);  // Create a 3x4 matrix instance for testing

    m.randomize();  // Apply randomization to the matrix elements

    // Print the matrix to visually verify the randomized values
    std::cout << "Printing randomized 3x4 matrix:" << std::endl;
    m.print(3);

    double* h_data = new double[3 * 4];  // Allocate host memory to copy matrix data from the device

    // Copy the matrix data from the device (GPU) to the host (CPU)
    cudaMemcpy(h_data, m.get_data(), 3 * 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that each element in the matrix is within the range [-0.5, 0.5]
    for (int i = 0; i < 3 * 4; ++i) {
        EXPECT_GE(h_data[i], -0.5);  // Verify that the value is greater than or equal to -0.5
        EXPECT_LE(h_data[i], 0.5);   // Verify that the value is less than or equal to 0.5
    }

    delete[] h_data;  // Free the allocated host memory
}
