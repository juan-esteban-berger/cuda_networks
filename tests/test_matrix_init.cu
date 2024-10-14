/**
 * @file test_matrix.cu
 * @brief Unit tests for the Matrix class.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>

/**
 * @class MatrixTest
 * @brief Test fixture for the Matrix class tests.
 */
class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Test that Matrix::initialize() sets all elements to zero.
 */
TEST_F(MatrixTest, InitializeToZero) {
    // Create a 3x4 matrix
    Matrix m(3, 4);
    
    // Initialize the matrix (should set all elements to 0)
    m.initialize();

    // Print the matrix to visually verify its contents
    std::cout << "Printing initialized matrix:" << std::endl;
    m.print();

    // Allocate host memory to verify the results
    double* h_data = new double[3 * 4];
    
    // Copy data from GPU to CPU for verification
    cudaMemcpy(h_data, m.get_data(), 3 * 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check that all elements are indeed 0
    for (int i = 0; i < 3 * 4; ++i) {
        EXPECT_DOUBLE_EQ(h_data[i], 0.0);
    }

    // Free the allocated host memory
    delete[] h_data;
}

/**
 * @brief Main function to run all tests.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return Integer 0 upon successful run, non-zero otherwise.
 */
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
