/**
 * @file neural_network_constructor.cu
 * @brief Implementation of the NeuralNetwork constructor.
 */

#include "neural_network.h"

NeuralNetwork::NeuralNetwork(int input_size,
                             int hidden_size,
                             int output_size)
    : input_size(input_size),
      hidden_size(hidden_size),
      output_size(output_size),
      W1(hidden_size, input_size),
      b1(hidden_size),
      W2(output_size, hidden_size),
      b2(output_size),
      A(input_size, 1),
      Z1(hidden_size, 1),
      A1(hidden_size, 1),
      Z2(output_size, 1),
      A2(output_size, 1),
      DZ2(output_size, 1),
      DW2(output_size, hidden_size),
      Db2(output_size),
      DZ1(hidden_size, 1),
      DW1(hidden_size, input_size),
      Db1(hidden_size) {
    // Initialize the neural network parameters
    initialize();
}
