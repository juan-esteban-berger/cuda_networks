/**
 * @file test_matrix_copy.cu
 * @brief Unit tests for the Matrix copy operations.
 */

#include <gtest/gtest.h>
#include "../../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

class MatrixCopyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MatrixCopyTest, CopyConstructor) {
    // Create original matrix
    Matrix original(3, 3);
    double h_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    cudaMemcpy(original.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original matrix
    std::cout << "Original matrix:" << std::endl;
    original.print(2);

    // Create copy using copy constructor
    Matrix copy(original);

    // Print copied matrix
    std::cout << "Copied matrix:" << std::endl;
    copy.print(2);

    // Verify dimensions and data
    EXPECT_EQ(original.get_rows(), copy.get_rows());
    EXPECT_EQ(original.get_cols(), copy.get_cols());
    EXPECT_NE(original.get_data(), copy.get_data());

    double* h_copy = new double[9];
    cudaMemcpy(h_copy, copy.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(h_data[i], h_copy[i]);
    }

    delete[] h_copy;
}

TEST_F(MatrixCopyTest, CopyAssignment) {
    // Create original matrix
    Matrix original(3, 3);
    double h_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    cudaMemcpy(original.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original matrix
    std::cout << "Original matrix:" << std::endl;
    original.print(2);

    // Create copy using copy assignment
    Matrix copy = original;

    // Print copied matrix
    std::cout << "Copied matrix:" << std::endl;
    copy.print(2);

    // Verify dimensions and data
    EXPECT_EQ(original.get_rows(), copy.get_rows());
    EXPECT_EQ(original.get_cols(), copy.get_cols());
    EXPECT_NE(original.get_data(), copy.get_data());

    double* h_copy = new double[9];
    cudaMemcpy(h_copy, copy.get_data(), 9 * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(h_data[i], h_copy[i]);
    }

    delete[] h_copy;
}

TEST_F(MatrixCopyTest, MoveConstructor) {
    // Create original matrix
    Matrix original(3, 3);
    double h_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    cudaMemcpy(original.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original matrix
    std::cout << "Original matrix before move:" << std::endl;
    original.print(2);

    double* original_data_ptr = original.get_data();
    Matrix moved(std::move(original));

    // Print moved matrix
    std::cout << "Moved matrix:" << std::endl;
    moved.print(2);

    // Verify move operation
    EXPECT_EQ(moved.get_rows(), 3);
    EXPECT_EQ(moved.get_cols(), 3);
    EXPECT_EQ(moved.get_data(), original_data_ptr);
    EXPECT_EQ(original.get_data(), nullptr);
}

TEST_F(MatrixCopyTest, MoveAssignment) {
    // Create original matrix
    Matrix original(3, 3);
    double h_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    cudaMemcpy(original.get_data(), h_data, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Print original matrix
    std::cout << "Original matrix before move assignment:" << std::endl;
    original.print(2);

    double* original_data_ptr = original.get_data();
    Matrix moved = std::move(original);

    // Print moved matrix
    std::cout << "Moved matrix after move assignment:" << std::endl;
    moved.print(2);

    // Verify move operation
    EXPECT_EQ(moved.get_rows(), 3);
    EXPECT_EQ(moved.get_cols(), 3);
    EXPECT_EQ(moved.get_data(), original_data_ptr);
    EXPECT_EQ(original.get_data(), nullptr);
}
