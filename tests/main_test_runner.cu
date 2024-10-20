/**
 * @file main_test_runner.cu
 * @brief Main test runner to execute all test cases.
 */
#include <gtest/gtest.h>

#include "linear_algebra/test_matrix_initialize.cu"
#include "linear_algebra/test_matrix_print.cu"
#include "linear_algebra/test_matrix_randomize.cu"
#include "linear_algebra/test_matrix_read_csv.cu"
#include "linear_algebra/test_matrix_read_csv_limited.cu"
#include "linear_algebra/test_matrix_preview_mnist_images.cu"
#include "linear_algebra/test_matrix_relu.cu"
#include "linear_algebra/test_matrix_relu_derivative.cu"
#include "linear_algebra/test_matrix_sigmoid.cu"
#include "linear_algebra/test_matrix_sigmoid_derivative.cu"
#include "linear_algebra/test_matrix_softmax.cu"
#include "linear_algebra/test_matrix_copy.cu"
#include "linear_algebra/test_matrix_multiply.cu"
#include "linear_algebra/test_matrix_multiply_elementwise.cu"
#include "linear_algebra/test_matrix_add_vector.cu"
#include "linear_algebra/test_matrix_subtract.cu"
#include "linear_algebra/test_matrix_sum.cu"
#include "linear_algebra/test_matrix_divide_scalar.cu"
#include "linear_algebra/test_matrix_multiply_scalar.cu"
#include "linear_algebra/test_matrix_argmax.cu"
#include "linear_algebra/test_matrix_transpose.cu"
#include "linear_algebra/test_matrix_select_batch.cu"

#include "linear_algebra/test_vector_initialize.cu"
#include "linear_algebra/test_vector_print.cu"
#include "linear_algebra/test_vector_randomize.cu"
#include "linear_algebra/test_vector_copy.cu"
#include "linear_algebra/test_vector_divide_scalar.cu"
#include "linear_algebra/test_vector_multiply_scalar.cu"
#include "linear_algebra/test_vector_subtract_scalar.cu"

#include "neural_network/test_neural_network_initialize.cu"
#include "neural_network/test_neural_network_forward.cu"
#include "neural_network/test_neural_network_backward.cu"
#include "neural_network/test_neural_network_update_params.cu"
#include "neural_network/test_neural_network_get_predictions.cu"
#include "neural_network/test_neural_network_get_accuracy.cu"
#include "neural_network/test_neural_network_gradient_descent.cu"

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
