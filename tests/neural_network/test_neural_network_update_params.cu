/**
 * @file test_neural_network_update_params.cu
 * @brief Unit tests for the NeuralNetwork::update_params method.
 */
#include <gtest/gtest.h>
#include "../../src/neural_network/neural_network.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class NeuralNetworkUpdateParamsTest
 * @brief Test fixture for the NeuralNetwork::update_params method tests.
 */
class NeuralNetworkUpdateParamsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that NeuralNetwork::update_params correctly updates network parameters.
 */
TEST_F(NeuralNetworkUpdateParamsTest, UpdateParamsTest) {
    // Create a neural network with 3 input neurons, 4 hidden neurons, and 2 output neurons
    NeuralNetwork nn(3, 4, 2);

    // Initialize the network
    nn.initialize();

    // Create input and label matrices
    Matrix X(3, 2);
    Matrix Y(2, 2);

    // Prepare test data
    double h_X[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    double h_Y[4] = {1.0, 0.0, 0.0, 1.0};

    // Copy test data to GPU
    cudaMemcpy(X.get_data(), h_X, 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.get_data(), h_Y, 4 * sizeof(double), cudaMemcpyHostToDevice);

    // Perform forward and backward propagation
    nn.forward(X);
    nn.backward(X, Y);

    // Update parameters
    double learning_rate = 0.1;
    nn.update_params(learning_rate);

    // Verify that gradients are not zero (indicating that update occurred)
    EXPECT_NE(nn.get_DW1().sum(), 0.0);
    EXPECT_NE(nn.get_DW2().sum(), 0.0);
    EXPECT_NE(nn.get_db1(), 0.0);
    EXPECT_NE(nn.get_db2(), 0.0);

    // Print gradient values
    std::cout << "DW1 sum: " << nn.get_DW1().sum() << std::endl;
    std::cout << "DW2 sum: " << nn.get_DW2().sum() << std::endl;
    std::cout << "db1: " << nn.get_db1() << std::endl;
    std::cout << "db2: " << nn.get_db2() << std::endl;
}
