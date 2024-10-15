/**
 * @file test_matrix_add_vector.cu
 * @brief Unit tests for the Matrix::add_vector method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include "../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixAddVectorTest
 * @brief Test fixture for the Matrix::add_vector method tests.
 */
class MatrixAddVectorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that Matrix::add_vector correctly adds a vector to each column of a matrix.
 */
TEST_F(MatrixAddVectorTest, AddVectorToMatrixCorrectly) {
    // Create test matrix and vector
    Matrix m(3, 2);
    Vector v(3);

    // Prepare test data
    double h_m[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double h_v[3] = {0.1, 0.2, 0.3};

    // Copy test data to GPU
    cudaMemcpy(m.get_data(), h_m, 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v.get_data(), h_v, 3 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original matrix and vector
    std::cout << "Original matrix:" << std::endl;
    m.print(2);
    std::cout << "Vector to add:" << std::endl;
    v.print(2);

    // Perform addition of vector to matrix
    m.add_vector(v);

    // Print result
    std::cout << "Matrix after adding vector:" << std::endl;
    m.print(2);

    // Copy result back to CPU for verification
    double* h_result = new double[6];
    cudaMemcpy(h_result, m.get_data(), 6 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define expected result
    double expected[6] = {1.1, 2.1, 3.2, 4.2, 5.3, 6.3};

    // Verify result
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-6);
    }

    // Clean up
    delete[] h_result;
}

/**
 * @test
 * @brief Verify that Matrix::add_vector throws an exception for incompatible dimensions.
 */
TEST_F(MatrixAddVectorTest, ThrowsExceptionForIncompatibleDimensions) {
    // Create matrix and vector with incompatible dimensions
    Matrix m(3, 2);
    Vector v(2);

    // Verify that an exception is thrown
    EXPECT_THROW(m.add_vector(v), std::invalid_argument);
}
