#include <cmath>

#include "linear_algebra.h"
#include "neural_network.h"

//////////////////////////////////////////////////////////////////
// Activation Function Classes
void Sigmoid::function(Matrix& Z) {
    for (int i = 0; i < Z.rows; i++) {
        for (int j = 0; j < Z.cols; j++) {
            double z = Z.getValues(i, j);
            Z.setValue(i, j, 1.0 / (1.0 + std::exp(-z)));
        }
    }
}

void Sigmoid::derivative(Matrix& Z) {
    for (int i = 0; i < Z.rows; i++) {
        for (int j = 0; j < Z.cols; j++) {
            double z = Z.getValues(i, j);
            double sigmoid_z = 1.0 / (1.0 + std::exp(-z));
            Z.setValue(i, j, sigmoid_z * (1.0 - sigmoid_z));
        }
    }
}

//////////////////////////////////////////////////////////////////
// Loss Function Classes
double CatCrossEntropy::function(Matrix& Y, Matrix& Y_hat) {
    double loss = 0.0;
    for (int i = 0; i < Y.rows; i++) {
        for (int j = 0; j < Y.cols; j++) {
            double y = Y.getValues(i, j);
            double y_hat = Y_hat.getValues(i, j);
            loss -= y * log(y_hat + 1e-8);
        }
    }
    return loss;
}

//////////////////////////////////////////////////////////////////
// Layer Class
Layer::Layer(int input_num, int output_num, std::string activation_func) {
    activation = activation_func;
    W = new Matrix(output_num, input_num);
    b = new Vector(output_num);
    
    random_matrix(W);
    random_vector(b);

    Z = nullptr;
    A = nullptr;
    dZ = nullptr;
    dW = nullptr;
    db = nullptr;
}

Layer::~Layer() {
    delete W;
    delete b;
    delete Z;
    delete A;
    delete dZ;
    delete dW;
    delete db;
}
