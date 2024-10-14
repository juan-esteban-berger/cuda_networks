/**
 * @file main_test_runner.cu
 * @brief Main test runner to execute all test cases.
 */
#include <gtest/gtest.h>

// #include "linear_algebra/test_matrix_initialize.cu"
// #include "linear_algebra/test_matrix_print.cu"
// #include "linear_algebra/test_matrix_randomize.cu"
// #include "linear_algebra/test_matrix_read_csv.cu"
// #include "linear_algebra/test_matrix_read_csv_limited.cu"
// #include "linear_algebra/test_matrix_preview_mnist_images.cu"
// #include "linear_algebra/test_matrix_relu.cu"
// #include "linear_algebra/test_matrix_softmax.cu"
// #include "linear_algebra/test_matrix_copy.cu"
// #include "linear_algebra/test_vector_initialize.cu"
// #include "linear_algebra/test_vector_print.cu"
// #include "linear_algebra/test_vector_randomize.cu"
#include "linear_algebra/test_vector_copy.cu"

/**
 * @brief Main function to run all tests.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return Integer 0 upon successful run, non-zero otherwise.
 */
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
