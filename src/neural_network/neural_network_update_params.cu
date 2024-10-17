/**
 * @file neural_network_update_params.cu
 * @brief Implementation of the NeuralNetwork::update_params method.
 */
#include "neural_network.h"
#include <iostream>

void NeuralNetwork::update_params(double learning_rate) {
    // Update weights for the first layer (W1)
    DW1.multiply_scalar(learning_rate);
    W1 = W1.subtract(DW1);
    // std::cout << "Updated W1:" << std::endl;
    // W1.print(4);

    // Update bias for the first layer (b1)
    b1.subtract_scalar(learning_rate * db1);
    // std::cout << "Updated b1:" << std::endl;
    // b1.print(4);

    // Update weights for the second layer (W2)
    DW2.multiply_scalar(learning_rate);
    W2 = W2.subtract(DW2);
    // std::cout << "Updated W2:" << std::endl;
    // W2.print(4);

    // Update bias for the second layer (b2)
    b2.subtract_scalar(learning_rate * db2);
    // std::cout << "Updated b2:" << std::endl;
    // b2.print(4);
}
