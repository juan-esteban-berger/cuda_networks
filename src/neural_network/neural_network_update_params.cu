/**
 * @file neural_network_update_params.cu
 * @brief Implementation of the NeuralNetwork::update_params method.
 */
#include "neural_network.h"

void NeuralNetwork::update_params(double learning_rate) {
    // Update weights for the first layer (W1)
    DW1.multiply_scalar(learning_rate);
    W1 = W1.subtract(DW1);

    // Update bias for the first layer (b1)
    b1.subtract_scalar(learning_rate * db1);

    // Update weights for the second layer (W2)
    DW2.multiply_scalar(learning_rate);
    W2 = W2.subtract(DW2);

    // Update bias for the second layer (b2)
    b2.subtract_scalar(learning_rate * db2);
}
