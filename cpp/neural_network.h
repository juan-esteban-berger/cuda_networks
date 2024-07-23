#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <cmath>

#include "utils.h"

////////////////////////////////////////////////////////////////////
// Activation Function Classes
class Sigmoid {
public:
    void function(DataFrame& Z);
    void derivative(DataFrame& Z);
};

class Softmax {
public:
    void function(DataFrame& Z);
};

////////////////////////////////////////////////////////////////////
// Loss Function Classes
class CatCrossEntropy {
public:
    double function(DataFrame& Y, DataFrame& Y_hat);
};

////////////////////////////////////////////////////////////////////
// Layer Class
class Layer {
public:
    DataFrame W;
    Series b;

    DataFrame Z;
    DataFrame A;

    DataFrame dZ;
    DataFrame dW;
    Series db;

    std::string activation;

    Layer(int input_num, int output_num, std::string activation_func);
};

////////////////////////////////////////////////////////////////////
// Neural Network Class
class NeuralNetwork {
private:
    Layer** layers;  // Pointer to an array of Layer pointers
    int numLayers;

public:
    NeuralNetwork(int numLayers);  // Constructor that specifies the number of layers
    void add_layer(Layer* layer, int index);  // Method to add layers
    ~NeuralNetwork();  // Destructor to clean up
};

#endif
