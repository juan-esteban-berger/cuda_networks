#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <string>
#include <vector>
#include <memory>

#include "linear_algebra.h"

//////////////////////////////////////////////////////////////////
// Activation Function Classes
class Sigmoid {
public:
    void function(Matrix& Z);
    void derivative(Matrix& Z);
};

class Softmax {
public:
    void function(Matrix& Z);
};

//////////////////////////////////////////////////////////////////
// Loss Function Classes
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

//////////////////////////////////////////////////////////////////
// Neural Network Class
class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();
    void add_layer(Layer* layer);
    Matrix* getOutput();
    void forward(Matrix& X);
    void backward(Matrix& X,
                  Matrix& Y,
                  std::string loss_func);
    void update_params(double learning_rate);
    double get_accuracy(Matrix& Y_true);
    void train(Matrix& X_train,
               Matrix& Y_train,
               int epochs,
               double learning_rate,
               std::string loss,
               std::string history_path);

    std::vector<Layer*> layers;
};

#endif // NEURAL_NETWORK_H
