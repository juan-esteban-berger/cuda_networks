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

////////////////////////////////////////////////////////////////////
// Neural Network Class

#endif
