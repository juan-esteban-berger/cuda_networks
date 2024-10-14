/**
 * @file test_matrix_print.cu
 * @brief Unit tests for the Matrix::print method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class MatrixPrintTest
 * @brief Test fixture for the Matrix::print method tests.
 */
class MatrixPrintTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Initializes and prints a matrix with random values.
     * 
     * This function randomizes the matrix contents using Matrix::randomize
     * and then calls the Matrix::print function to display it.
     *
     * @param m Reference to the matrix to randomize and print.
     * @param label A string label for identifying the printed matrix in output.
     */
    void randomizeAndPrint(Matrix &m, const std::string &label) {
        m.randomize();  // Fill the matrix with random values

        // Print the matrix with a label to identify the size in output
        std::cout << "Printing randomized " << label << " matrix:\n";
        m.print(2);
    }
};

/**
 * @test
 * @brief Test Matrix::print() with a 5x5 random matrix.
 */
TEST_F(MatrixPrintTest, Print5x5Matrix) {
    Matrix m(5, 5);
    randomizeAndPrint(m, "5x5");
}

/**
 * @test
 * @brief Test Matrix::print() with a 5x20 random matrix.
 */
TEST_F(MatrixPrintTest, Print5x20Matrix) {
    Matrix m(5, 20);
    randomizeAndPrint(m, "5x20");
}

/**
 * @test
 * @brief Test Matrix::print() with a 20x5 random matrix.
 */
TEST_F(MatrixPrintTest, Print20x5Matrix) {
    Matrix m(20, 5);
    randomizeAndPrint(m, "20x5");
}

/**
 * @test
 * @brief Test Matrix::print() with a 20x20 random matrix.
 */
TEST_F(MatrixPrintTest, Print20x20Matrix) {
    Matrix m(20, 20);
    randomizeAndPrint(m, "20x20");
}

