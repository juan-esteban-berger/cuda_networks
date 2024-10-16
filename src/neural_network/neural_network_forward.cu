/**
 * @file neural_network_forward.cu
 * @brief Implementation of the NeuralNetwork::forward method.
 */

#include "neural_network.h"

void NeuralNetwork::forward(const Matrix& X) {
    // Store the input matrix
    A = X.copy();

    // Compute the pre-activation of the hidden layer
    Z1 = W1.multiply(A);

    // Add biases to the pre-activation
    Z1.add_vector(b1);

    // Apply ReLU activation to the hidden layer
    A1 = Z1.relu();

    // Compute the pre-activation of the output layer
    Z2 = W2.multiply(A1);

    // Add biases to the pre-activation
    Z2.add_vector(b2);

    // Apply softmax activation to the output layer
    A2 = Z2.softmax();
}
