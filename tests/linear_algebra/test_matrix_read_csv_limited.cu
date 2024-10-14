/**
 * @file test_matrix_read_csv_limited.cu
 * @brief Unit tests for the Matrix::read_csv_limited method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

/**
 * @class MatrixReadCSVLimitedTest
 * @brief Test fixture for the Matrix::read_csv_limited method tests.
 */
class MatrixReadCSVLimitedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary CSV file for testing
        std::ofstream testFile("test_matrix_large.csv");
        for (int i = 0; i < 5; ++i) {
            testFile << i*5+1 << "," << i*5+2 << "," << i*5+3 << "," << i*5+4 << "," << i*5+5 << "\n";
        }
        testFile.close();
    }

    void TearDown() override {
        // Remove the temporary CSV file
        std::remove("test_matrix_large.csv");
    }
};

/**
 * @test
 * @brief Verify that Matrix::read_csv_limited correctly reads a subset of data from a CSV file.
 *
 * This test creates a matrix, reads a subset of data from a CSV file into it,
 * and confirms that all elements are correctly loaded.
 */
TEST_F(MatrixReadCSVLimitedTest, ReadCSVLimitedCorrectly) {
    // Create a 2x5 matrix to store a subset of the test CSV file
    Matrix m(2, 5);

    // Read rows 2-4 (0-based index 1-3) from the CSV file into the matrix
    m.read_csv_limited("test_matrix_large.csv", 1, 3, 5, 5);

    // Print the matrix to verify loaded data
    std::cout << "Matrix loaded from CSV (rows 2-3):\n";
    m.print(1);

    // Allocate host memory to verify the results
    double* h_data = new double[2 * 5];
    cudaMemcpy(h_data, m.get_data(), 2 * 5 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are correctly loaded
    double expected_data[10] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(h_data[i], expected_data[i]);
    }

    // Free the allocated host memory
    delete[] h_data;
}

/**
 * @test
 * @brief Verify that Matrix::read_csv_limited throws an exception for invalid row ranges.
 */
TEST_F(MatrixReadCSVLimitedTest, ThrowsExceptionForInvalidRange) {
    Matrix m(2, 5);
    EXPECT_THROW(m.read_csv_limited("test_matrix_large.csv", 3, 2, 5, 5), std::runtime_error);
    EXPECT_THROW(m.read_csv_limited("test_matrix_large.csv", -1, 2, 5, 5), std::runtime_error);
    EXPECT_THROW(m.read_csv_limited("test_matrix_large.csv", 0, 6, 5, 5), std::runtime_error);
}

/**
 * @test
 * @brief Verify that Matrix::read_csv_limited throws an exception when matrix dimensions don't match the specified range.
 */
TEST_F(MatrixReadCSVLimitedTest, ThrowsExceptionForMismatchedDimensions) {
    Matrix m(3, 5);  // Create a 3x5 matrix
    EXPECT_THROW(m.read_csv_limited("test_matrix_large.csv", 1, 3, 5, 5), std::runtime_error);
}
