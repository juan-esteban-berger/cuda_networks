/**
 * @file test_neural_network_get_predictions.cu
 * @brief Unit tests for the NeuralNetwork::get_predictions method.
 */
#include <gtest/gtest.h>
#include "../../src/neural_network/neural_network.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class NeuralNetworkGetPredictionsTest
 * @brief Test fixture for the NeuralNetwork::get_predictions method tests.
 */
class NeuralNetworkGetPredictionsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that NeuralNetwork::get_predictions correctly computes predictions.
 */
TEST_F(NeuralNetworkGetPredictionsTest, GetPredictionsTest) {
    // Create a neural network with 3 input neurons, 4 hidden neurons, and 2 output neurons
    NeuralNetwork nn(3, 4, 2);

    // Initialize the network
    nn.initialize();

    // Create an input matrix (batch size of 3)
    Matrix X(3, 3);
    double h_X[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    cudaMemcpy(X.get_data(), h_X, 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Perform forward propagation
    nn.forward(X);

    // Get A2 matrix dimensions
    auto A2_dims = nn.get_A2_dimensions();
    
    // Create A2 matrix and copy data
    Matrix A2(A2_dims.first, A2_dims.second);
    cudaMemcpy(A2.get_data(), nn.get_A2_data(), A2_dims.first * A2_dims.second * sizeof(double), cudaMemcpyDeviceToDevice);

    // Print A2 matrix
    std::cout << "A2 matrix:" << std::endl;
    A2.print(4);

    // Get predictions
    Vector predictions = nn.get_predictions();

    // Verify predictions dimensions
    EXPECT_EQ(predictions.get_rows(), 3);

    // Print predictions
    std::cout << "Predictions:" << std::endl;
    predictions.print(0);

    // Verify that predictions are either 0 or 1 (since we have 2 output neurons)
    double* h_predictions = new double[3];
    cudaMemcpy(h_predictions, predictions.get_data(), 3 * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify predictions match A2 argmax
    double* h_A2 = new double[A2_dims.first * A2_dims.second];
    cudaMemcpy(h_A2, A2.get_data(), A2_dims.first * A2_dims.second * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 3; ++i) {
        // Check if prediction matches the index of the maximum value in A2
        int max_index = (h_A2[i] > h_A2[i + 3]) ? 0 : 1;
        EXPECT_EQ(h_predictions[i], max_index);
        
        // Additional check to ensure predictions are 0 or 1
        EXPECT_TRUE(h_predictions[i] == 0 || h_predictions[i] == 1);
    }

    // Clean up
    delete[] h_predictions;
    delete[] h_A2;
}
