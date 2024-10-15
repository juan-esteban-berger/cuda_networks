/**
 * @file test_matrix_argmax.cu
 * @brief Unit tests for the Matrix::argmax method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include "../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixArgmaxTest
 * @brief Test fixture for the Matrix::argmax method tests.
 */
class MatrixArgmaxTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::argmax correctly computes the column-wise argmax.
 *
 * This test creates a matrix with known values,
 * computes the column-wise argmax, and confirms that the results are correct.
 */
TEST_F(MatrixArgmaxTest, ComputeArgmaxCorrectly) {
    // Create a 3x4 matrix with known values
    Matrix m(3, 4);
    double h_data[12] = {1.0, 3.0, 2.0, 4.0,
                         5.0, 2.0, 6.0, 1.0,
                         3.0, 7.0, 4.0, 2.0};
    cudaMemcpy(m.get_data(), h_data, 12 * sizeof(double), cudaMemcpyHostToDevice);

    // Compute the column-wise argmax
    Vector result = m.argmax();

    // Print the original matrix and result vector
    std::cout << "Original matrix:" << std::endl;
    m.print(2);
    std::cout << "Column-wise argmax:" << std::endl;
    result.print(0);

    // Allocate host memory to verify the results
    double h_result[4];
    cudaMemcpy(h_result, result.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define the expected result
    double expected[4] = {1, 2, 1, 0};

    // Check that all elements are correct
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(h_result[i], expected[i]);
    }
}
