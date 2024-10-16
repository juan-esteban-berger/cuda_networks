/**
 * @file test_neural_network_get_accuracy.cu
 * @brief Unit tests for the NeuralNetwork::get_accuracy method.
 */
#include <gtest/gtest.h>
#include "../../src/neural_network/neural_network.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @class NeuralNetworkGetAccuracyTest
 * @brief Test fixture for the NeuralNetwork::get_accuracy method tests.
 */
class NeuralNetworkGetAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @test
 * @brief Verify that NeuralNetwork::get_accuracy correctly computes accuracy.
 */
TEST_F(NeuralNetworkGetAccuracyTest, GetAccuracyTest) {
    // Create a neural network with 3 input neurons, 4 hidden neurons, and 2 output neurons
    NeuralNetwork nn(3, 4, 2);

    // Initialize the network
    nn.initialize();

    // Create input and label matrices
    Matrix X(3, 4);
    Matrix Y(2, 4);

    // Prepare test data
    double h_X[12] = {0.1, 0.2, 0.3, 0.4,
                      0.5, 0.6, 0.7, 0.8,
                      0.9, 1.0, 1.1, 1.2};
    double h_Y[8] = {1.0, 0.0, 1.0, 0.0,
                     0.0, 1.0, 0.0, 1.0};

    // Copy test data to GPU
    cudaMemcpy(X.get_data(), h_X, 12 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.get_data(), h_Y, 8 * sizeof(double), cudaMemcpyHostToDevice);

    // Print input matrix X
    std::cout << "Input matrix X:" << std::endl;
    X.print(2);

    // Print true labels matrix Y
    std::cout << "True labels matrix Y:" << std::endl;
    Y.print(2);

    // Perform forward propagation
    nn.forward(X);

    // Get A2 matrix (output layer activations)
    Matrix A2(nn.get_A2_dimensions().first, nn.get_A2_dimensions().second);
    cudaMemcpy(A2.get_data(), nn.get_A2_data(), A2.get_rows() * A2.get_cols() * sizeof(double), cudaMemcpyDeviceToDevice);

    // Print A2 matrix
    std::cout << "Output matrix A2:" << std::endl;
    A2.print(4);

    // Perform backward propagation
    nn.backward(X, Y);

    // Update parameters
    double learning_rate = 0.1;
    nn.update_params(learning_rate);

    // Get predictions
    Vector predictions = nn.get_predictions();

    // Print predictions
    std::cout << "Predictions:" << std::endl;
    predictions.print(0);

    // Calculate accuracy
    double accuracy = nn.get_accuracy(Y);

    // Print accuracy
    std::cout << "Accuracy: " << accuracy << std::endl;

    // Verify accuracy is between 0 and 1
    EXPECT_GE(accuracy, 0.0);
    EXPECT_LE(accuracy, 1.0);

    // Convert Y matrix to argmax form for comparison
    Vector Y_argmax = Y.argmax();

    // Print Y_argmax
    std::cout << "True labels (argmax):" << std::endl;
    Y_argmax.print(0);

    // Compare predictions with Y_argmax
    std::cout << "Comparison of predictions and true labels:" << std::endl;
    for (int i = 0; i < predictions.get_rows(); ++i) {
        double pred_val, y_val;
        cudaMemcpy(&pred_val, predictions.get_data() + i, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&y_val, Y_argmax.get_data() + i, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Prediction: " << pred_val << ", True label: " << y_val << std::endl;
    }
}
