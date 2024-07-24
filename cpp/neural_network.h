#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

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

#endif // NEURAL_NETWORK_H
