/**
 * @file neural_network_get_predictions.cu
 * @brief Implementation of the NeuralNetwork::get_predictions method.
 */
#include "neural_network.h"

Vector NeuralNetwork::get_predictions() const {
    // Get the argmax of A2 along axis 0 (column-wise)
    return A2.argmax();
}
