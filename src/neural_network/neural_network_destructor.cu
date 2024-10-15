/**
 * @file neural_network_destructor.cu
 * @brief Implementation of the NeuralNetwork destructor.
 */

#include "neural_network.h"

NeuralNetwork::~NeuralNetwork() {
    // The destructor for Matrix and Vector objects will be called automatically
    // to free the GPU memory
}
