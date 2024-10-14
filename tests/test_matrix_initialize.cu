/**
 * @file test_matrix_init.cu
 * @brief Unit tests for the Matrix::initialize method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixInitTest
 * @brief Test fixture for the Matrix::initialize tests.
 */
class MatrixInitTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Test that Matrix::initialize() sets all elements to zero.
 */
TEST_F(MatrixInitTest, InitializeToZero) {
    Matrix m(3, 4);  // Create a 3x4 matrix
    m.initialize();  // Initialize the matrix (should set all elements to 0)

    // Print the matrix to verify initialization
    std::cout << "Printing initialized 3x4 matrix:\n";
    m.print(2);

    // Allocate host memory to verify the results
    double* h_data = new double[3 * 4];
    cudaMemcpy(h_data, m.get_data(), 3 * 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are indeed 0
    for (int i = 0; i < 3 * 4; ++i) {
        EXPECT_DOUBLE_EQ(h_data[i], 0.0);
    }

    // Free the allocated host memory
    delete[] h_data;
}
