#include <iostream>
#include <cmath>
#include <string>
#include <vector>

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

void Softmax::function(Matrix& Z) {
    // Loop through each column
    for (int j = 0; j < Z.cols; j++) {
        // Find max value in the column
        double max_val = Z.getValues(0, j);
        for (int i = 1; i < Z.rows; i++) {
            double temp_val = Z.getValues(i, j);
            if (temp_val > max_val) {
                max_val = temp_val;
            }
        }

        // Compute exp(Z - max)
        double sum = 0.0;
        for (int i = 0; i < Z.rows; i++) {
            double exp_val = std::exp(Z.getValues(i, j) - max_val);
            Z.setValue(i, j, exp_val);
        }

        // Compute sum(exp(Z - max))
        for (int i = 0; i < Z.rows; i++) {
            sum += Z.getValues(i, j);
        }

        // Divide by sum
        for (int i = 0; i < Z.rows; i++) {
            Z.setValue(i, j, Z.getValues(i, j) / (sum + 1e-8));
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

//////////////////////////////////////////////////////////////////
// Neural Network Class
NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::~NeuralNetwork() {
    for (Layer* layer : layers) {
        delete layer;
    }
}

void NeuralNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
}

Matrix* NeuralNetwork::getOutput() {
    return layers.back()->A;
}

void NeuralNetwork::forward(Matrix& X) {
    // Initialize Matrix A
    Matrix A(X.rows, X.cols);

    // Copy from Matrix X into Matrix A
    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            A.setValue(i, j, X.getValues(i, j));
        }
    }

    // Initialize Sigmoid object
    Sigmoid sigmoid;
    // Initialize Softmax object
    Softmax softmax;
    
    // Loop through each layer
    for (Layer* layer : layers) {
        // Multiply weights by input matrix
        Matrix Z_temp = matmul(*layer->W, A);

        // Add bias to Z matrix
        Matrix Z = Z_temp + *layer->b;
        
        // Create a new Matrix for Z and copy values
        layer->Z = new Matrix(Z.rows, Z.cols);
        for (int i = 0; i < Z.rows; i++) {
            for (int j = 0; j < Z.cols; j++) {
                layer->Z->setValue(i, j, Z.getValues(i, j));
            }
        }
        
        // if (layer->activation == "Sigmoid")
        if (layer->activation == "Sigmoid") {
            // Compute Sigmoid function
            sigmoid.function(Z);

        } else if (layer->activation == "Softmax") {
            // Compute Softmax function
            softmax.function(Z);

        }

        // Create a new matrix copy values
        layer->A = new Matrix(Z.rows, Z.cols);
        for (int i = 0; i < Z.rows; i++) {
            for (int j = 0; j < Z.cols; j++) {
                layer->A->setValue(i, j, Z.getValues(i, j));
            }
        }

        // Update A for the next iteration
        Matrix A_temp(layer->A->rows, layer->A->cols);
        for (int i = 0; i < layer->A->rows; i++) {
            for (int j = 0; j < layer->A->cols; j++) {
                A_temp.setValue(i, j, layer->A->getValues(i, j));
            }
        }

        // Manually copy A_temp to A
        for (int i = 0; i < A.rows; i++) {
            delete[] A.data[i];
        }
        delete[] A.data;

        A.rows = A_temp.rows;
        A.cols = A_temp.cols;
        A.data = new double*[A.rows];
        for (int i = 0; i < A.rows; i++) {
            A.data[i] = new double[A.cols];
            for (int j = 0; j < A.cols; j++) {
                A.setValue(i, j, A_temp.getValues(i, j));
            }
        }

    }
}

