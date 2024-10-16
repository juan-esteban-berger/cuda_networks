/**
 * @file test_neural_network_forward.cu
 * @brief Unit tests for the NeuralNetwork::forward method.
 */

#include <gtest/gtest.h>
#include "../../src/neural_network/neural_network.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

/**
 * @class NeuralNetworkForwardTest
 * @brief Test fixture for the NeuralNetwork::forward method tests.
 */
class NeuralNetworkForwardTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Helper function to check if two doubles are approximately equal
     * @param a First value
     * @param b Second value
     * @param epsilon Tolerance for comparison
     * @return true if values are approximately equal, false otherwise
     */
    bool isApproximatelyEqual(double a, double b, double epsilon = 1e-6) {
        return std::fabs(a - b) < epsilon;
    }
};

/**
 * @test
 * @brief Verify that NeuralNetwork::forward correctly performs forward propagation.
 */
TEST_F(NeuralNetworkForwardTest, ForwardPropagationTest) {
    // Create a neural network with 3 input neurons, 4 hidden neurons, and 2 output neurons
    NeuralNetwork nn(3, 4, 2);

    // Initialize the network
    nn.initialize();

    // Create an input matrix (batch size of 2)
    Matrix X(3, 2);
    double h_X[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    cudaMemcpy(X.get_data(), h_X, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Print the input matrix
    std::cout << "Input matrix:" << std::endl;
    X.print(2);

    // Perform forward propagation
    nn.forward(X);

    // Print and verify the output matrix (A2)
    std::cout << "Output matrix (A2):" << std::endl;
    Matrix A2(nn.get_A2_dimensions().first, nn.get_A2_dimensions().second);
    cudaMemcpy(A2.get_data(), nn.get_A2_data(), A2.get_rows() * A2.get_cols() * sizeof(double), cudaMemcpyDeviceToDevice);
    A2.print(4);

    // Verify output dimensions
    EXPECT_EQ(A2.get_rows(), 2);
    EXPECT_EQ(A2.get_cols(), 2);

    // Verify output values are probabilities (sum to 1 for each sample)
    double* h_A2 = new double[4];
    cudaMemcpy(h_A2, A2.get_data(), 4 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; ++i) {
        double sum = h_A2[i] + h_A2[i + 2];
        EXPECT_TRUE(isApproximatelyEqual(sum, 1.0, 1e-6));
    }

    // Print and verify intermediate matrices
    std::cout << "Z1 matrix:" << std::endl;
    Matrix Z1(nn.get_Z1_dimensions().first, nn.get_Z1_dimensions().second);
    cudaMemcpy(Z1.get_data(), nn.get_Z1_data(), Z1.get_rows() * Z1.get_cols() * sizeof(double), cudaMemcpyDeviceToDevice);
    Z1.print(4);

    std::cout << "A1 matrix:" << std::endl;
    Matrix A1(nn.get_A1_dimensions().first, nn.get_A1_dimensions().second);
    cudaMemcpy(A1.get_data(), nn.get_A1_data(), A1.get_rows() * A1.get_cols() * sizeof(double), cudaMemcpyDeviceToDevice);
    A1.print(4);

    std::cout << "Z2 matrix:" << std::endl;
    Matrix Z2(nn.get_Z2_dimensions().first, nn.get_Z2_dimensions().second);
    cudaMemcpy(Z2.get_data(), nn.get_Z2_data(), Z2.get_rows() * Z2.get_cols() * sizeof(double), cudaMemcpyDeviceToDevice);
    Z2.print(4);

    // Verify A1 values are non-negative (ReLU output)
    double* h_A1 = new double[A1.get_rows() * A1.get_cols()];
    cudaMemcpy(h_A1, A1.get_data(), A1.get_rows() * A1.get_cols() * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < A1.get_rows() * A1.get_cols(); ++i) {
        EXPECT_GE(h_A1[i], 0.0);
    }

    // Clean up
    delete[] h_A2;
    delete[] h_A1;
}
