/**
 * @file neural_network_forward.cu
 * @brief Implementation of the NeuralNetwork::forward method.
 */
#include "neural_network.h"
#include <iostream>

void NeuralNetwork::forward(const Matrix& X) {
    // Store the input matrix
    A = X.copy();
    // std::cout << "Input matrix A:" << std::endl;
    // A.print(4);

    // Compute the pre-activation of the hidden layer
    Z1 = W1.multiply(A);
    // std::cout << "Pre-activation of hidden layer Z1:" << std::endl;
    // Z1.print(4);

    // Add biases to the pre-activation
    Z1.add_vector(b1);
    // std::cout << "Z1 after adding biases:" << std::endl;
    // Z1.print(4);

    // Apply ReLU activation to the hidden layer
    A1 = Z1.relu();
    // std::cout << "Activation of hidden layer A1:" << std::endl;
    // A1.print(4);

    // Compute the pre-activation of the output layer
    Z2 = W2.multiply(A1);
    // std::cout << "Pre-activation of output layer Z2:" << std::endl;
    // Z2.print(4);

    // Add biases to the pre-activation
    Z2.add_vector(b2);
    // std::cout << "Z2 after adding biases:" << std::endl;
    // Z2.print(4);

    // Apply softmax activation to the output layer
    A2 = Z2.softmax();
    // std::cout << "Activation of output layer A2:" << std::endl;
    // A2.print(4);
}
