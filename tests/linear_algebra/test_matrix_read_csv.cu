/**
 * @file test_matrix_read_csv.cu
 * @brief Unit tests for the Matrix::read_csv method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

/**
 * @class MatrixReadCSVTest
 * @brief Test fixture for the Matrix::read_csv method tests.
 */
class MatrixReadCSVTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary CSV file with known data for testing
        std::ofstream testFile("test_matrix.csv");
        testFile << "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n";
        testFile.close();
    }

    void TearDown() override {
        // Clean up by removing the temporary CSV file
        std::remove("test_matrix.csv");
    }
};

/**
 * @test
 * @brief Verify that Matrix::read_csv correctly reads data from a CSV file.
 *
 * This test creates a matrix, reads data from a CSV file into it,
 * and confirms that all elements are correctly loaded.
 */
TEST_F(MatrixReadCSVTest, ReadCSVCorrectly) {
    // Create a 3x3 matrix to match the dimensions of our test CSV file
    Matrix m(3, 3);

    // Read the CSV file into the matrix
    m.read_csv("test_matrix.csv");

    // Print the matrix to visually verify loaded data
    std::cout << "Matrix loaded from CSV:\n";
    m.print(1);

    // Allocate host memory to verify the results
    double* h_data = new double[3 * 3];

    // Copy data from device (GPU) to host (CPU) for verification
    cudaMemcpy(h_data, m.get_data(), 3 * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    // Define the expected data based on our test CSV file
    double expected_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Check that all elements are correctly loaded
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(h_data[i], expected_data[i]);
    }

    // Free the allocated host memory
    delete[] h_data;
}

/**
 * @test
 * @brief Verify that Matrix::read_csv throws an exception for non-existent files.
 */
TEST_F(MatrixReadCSVTest, ThrowsExceptionForNonExistentFile) {
    Matrix m(3, 3);
    // Attempt to read a non-existent file, expect a runtime_error
    EXPECT_THROW(m.read_csv("non_existent_file.csv"), std::runtime_error);
}

/**
 * @test
 * @brief Verify that Matrix::read_csv throws an exception when CSV data doesn't match matrix dimensions.
 */
TEST_F(MatrixReadCSVTest, ThrowsExceptionForMismatchedDimensions) {
    // Create a 2x2 matrix, which doesn't match our 3x3 test CSV
    Matrix m(2, 2);
    // Attempt to read 3x3 data into 2x2 matrix, expect a runtime_error
    EXPECT_THROW(m.read_csv("test_matrix.csv"), std::runtime_error);
}
