/**
 * @file test_neural_network_backward.cu
 * @brief Unit tests for the NeuralNetwork::backward method.
 */

#include <gtest/gtest.h>
#include "../../src/neural_network/neural_network.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class NeuralNetworkBackwardTest
 * @brief Test fixture for the NeuralNetwork::backward method tests.
 */
class NeuralNetworkBackwardTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that NeuralNetwork::backward computes gradients with correct dimensions.
 */
TEST_F(NeuralNetworkBackwardTest, BackwardPropagationBasicTest) {
    // Create a neural network with 3 input neurons, 4 hidden neurons, and 2 output neurons
    NeuralNetwork nn(3, 4, 2);

    // Create input and label matrices
    Matrix X(3, 2);
    Matrix Y(2, 2);

    // Prepare test data
    double h_X[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    double h_Y[4] = {1.0, 0.0, 0.0, 1.0};

    // Copy test data to GPU
    cudaMemcpy(X.get_data(), h_X, 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.get_data(), h_Y, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Print input matrices
    std::cout << "Input matrix X:" << std::endl;
    X.print(2);
    std::cout << "Label matrix Y:" << std::endl;
    Y.print(2);

    // Perform forward and backward propagation
    nn.forward(X);
    nn.backward(X, Y);

    // Get computed gradients
    Matrix DW1 = nn.get_DW1();
    Matrix DW2 = nn.get_DW2();
    double db1 = nn.get_db1();
    double db2 = nn.get_db2();

    // Print gradient dimensions and scalar values
    std::cout << "DW1 dimensions: " << DW1.get_rows() << "x" << DW1.get_cols() << std::endl;
    std::cout << "DW2 dimensions: " << DW2.get_rows() << "x" << DW2.get_cols() << std::endl;
    std::cout << "db1 value: " << db1 << std::endl;
    std::cout << "db2 value: " << db2 << std::endl;

    // Verify gradient dimensions
    EXPECT_EQ(DW1.get_rows(), 4);
    EXPECT_EQ(DW1.get_cols(), 3);
    EXPECT_EQ(DW2.get_rows(), 2);
    EXPECT_EQ(DW2.get_cols(), 4);

    // Verify that gradients are not all zero
    EXPECT_NE(DW1.sum(), 0.0);
    EXPECT_NE(DW2.sum(), 0.0);
    EXPECT_NE(db1, 0.0);
    EXPECT_NE(db2, 0.0);
}
