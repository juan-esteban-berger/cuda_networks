/**
 * @file neural_network_initialize.cu
 * @brief Implementation of the NeuralNetwork::initialize method.
 */

#include "neural_network.h"
#include <cmath>

void NeuralNetwork::initialize() {
    // Initialize W1 with random values
    W1.randomize();
    // Scale W1 by sqrt(2.0 / input_size) for better initial performance
    W1.multiply_scalar(std::sqrt(2.0 / input_size));

    // Initialize b1 with random values
    b1.randomize();
    // Scale b1 by 0.01 to keep initial values small
    b1.multiply_scalar(0.01);

    // Initialize W2 with random values
    W2.randomize();
    // Scale W2 by sqrt(2.0 / hidden_size) for better initial performance
    W2.multiply_scalar(std::sqrt(2.0 / hidden_size));

    // Initialize b2 with random values
    b2.randomize();
    // Scale b2 by 0.01 to keep initial values small
    b2.multiply_scalar(0.01);

    // Initialize other matrices with zeros
    A.initialize();
    Z1.initialize();
    A1.initialize();
    Z2.initialize();
    A2.initialize();
    DZ2.initialize();
    DW2.initialize();
    DZ1.initialize();
    DW1.initialize();

    // Initialize scalar gradients to zero
    db1 = 0.0;
    db2 = 0.0;
}
