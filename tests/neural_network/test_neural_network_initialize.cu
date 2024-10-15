/**
 * @file test_neural_network_initialize.cu
 * @brief Unit tests for the NeuralNetwork::initialize method.
 */

#include <gtest/gtest.h>
#include "../../src/neural_network/neural_network.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

/**
 * @brief Helper function to check if data contains non-zero values
 * @param data Pointer to the data on the device
 * @param size Number of elements to check
 * @return true if the data contains non-zero values, false otherwise
 */
bool containsNonZero(const double* data, int size) {
    // Allocate host memory
    double* h_data = new double[size];
    
    // Copy data from device to host
    cudaMemcpy(h_data, data, size * sizeof(double), cudaMemcpyDeviceToHost);
    
    bool hasNonZero = false;
    // Check each element
    for (int i = 0; i < size; ++i) {
        if (h_data[i] != 0.0) {
            hasNonZero = true;
            break;
        }
    }
    
    // Free host memory
    delete[] h_data;
    return hasNonZero;
}

/**
 * @brief Test case for NeuralNetwork::initialize method
 */
TEST(NeuralNetworkInitializeTest, InitializeNetworkCorrectly) {
    // Create a neural network with 784 input neurons, 10 hidden neurons, and 10 output neurons
    NeuralNetwork nn(784, 10, 10);

    // Get dimensions and sizes
    auto W1_dim = nn.get_W1_dimensions();
    auto W2_dim = nn.get_W2_dimensions();
    int b1_size = nn.get_b1_size();
    int b2_size = nn.get_b2_size();

    // Verify that W1 and W2 contain non-zero values
    EXPECT_TRUE(containsNonZero(nn.get_W1_data(), W1_dim.first * W1_dim.second));
    EXPECT_TRUE(containsNonZero(nn.get_W2_data(), W2_dim.first * W2_dim.second));

    // Verify that b1 and b2 contain non-zero values
    EXPECT_TRUE(containsNonZero(nn.get_b1_data(), b1_size));
    EXPECT_TRUE(containsNonZero(nn.get_b2_data(), b2_size));

    // Print W1 for visual verification
    std::cout << "W1 (first few elements):" << std::endl;
    Matrix W1(W1_dim.first, W1_dim.second);
    cudaMemcpy(W1.get_data(), nn.get_W1_data(), W1_dim.first * W1_dim.second * sizeof(double), cudaMemcpyDeviceToDevice);
    W1.print(4);

    // Print W2 for visual verification
    std::cout << "W2 (first few elements):" << std::endl;
    Matrix W2(W2_dim.first, W2_dim.second);
    cudaMemcpy(W2.get_data(), nn.get_W2_data(), W2_dim.first * W2_dim.second * sizeof(double), cudaMemcpyDeviceToDevice);
    W2.print(4);

    // Print b1 for visual verification
    std::cout << "b1:" << std::endl;
    Vector b1(b1_size);
    cudaMemcpy(b1.get_data(), nn.get_b1_data(), b1_size * sizeof(double), cudaMemcpyDeviceToDevice);
    b1.print(4);

    // Print b2 for visual verification
    std::cout << "b2:" << std::endl;
    Vector b2(b2_size);
    cudaMemcpy(b2.get_data(), nn.get_b2_data(), b2_size * sizeof(double), cudaMemcpyDeviceToDevice);
    b2.print(4);
}
