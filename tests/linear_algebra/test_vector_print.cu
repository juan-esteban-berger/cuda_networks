/**
 * @file test_vector_print.cu
 * @brief Unit tests for the Vector::print method.
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/vector.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class VectorPrintTest
 * @brief Test fixture for the Vector::print method tests.
 */
class VectorPrintTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Initializes and prints a vector with random values.
     * 
     * This function randomizes the vector contents using Vector::randomize
     * and then calls the Vector::print function to display it.
     *
     * @param v Reference to the vector to randomize and print.
     * @param label A string label for identifying the printed vector in output.
     */
    void randomizeAndPrint(Vector &v, const std::string &label) {
        v.randomize();  // Fill the vector with random values

        // Print the vector with a label to identify the size in output
        std::cout << "Printing randomized " << label << " vector:\n";
        v.print(2);
    }
};

/**
 * @test
 * @brief Test Vector::print() with a 5-element random vector.
 */
TEST_F(VectorPrintTest, Print5ElementVector) {
    Vector v(5);
    randomizeAndPrint(v, "5-element");
}

/**
 * @test
 * @brief Test Vector::print() with a 20-element random vector.
 */
TEST_F(VectorPrintTest, Print20ElementVector) {
    Vector v(20);
    randomizeAndPrint(v, "20-element");
}

/**
 * @test
 * @brief Test Vector::print() with a 50-element random vector.
 */
TEST_F(VectorPrintTest, Print50ElementVector) {
    Vector v(50);
    randomizeAndPrint(v, "50-element");
}
