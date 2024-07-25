#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <string>

#include "linear_algebra.h"

//////////////////////////////////////////////////////////////////
// Activation Function Classes
class Sigmoid {
public:
    void function(Matrix& Z);
    void derivative(Matrix& Z);
};

class CatCrossEntropy {
public:
    double function(Matrix& Y, Matrix& Y_hat);
};

//////////////////////////////////////////////////////////////////
// Layer Class
class Layer {
public:
    Matrix* W;
    Vector* b;
    Matrix* Z;
    Matrix* A;
    Matrix* dZ;
    Matrix* dW;
    Vector* db;
    std::string activation;

    Layer(int input_num, int output_num, std::string activation_func);
    ~Layer();
};

#endif // NEURAL_NETWORK_H
