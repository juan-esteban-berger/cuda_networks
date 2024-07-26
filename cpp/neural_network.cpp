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
    std::cout << "Entering forward function\n";
    std::cout << "Input matrix X:\n";
    preview_matrix(&X, 4);

    Matrix* A = &X;

    Sigmoid sigmoid;
    Softmax softmax;

    std::cout << "Number of layers: " << layers.size() << "\n";

    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "\nProcessing layer " << i << "\n";
        Layer* layer = layers[i];

        std::cout << "Layer weights:\n";
        preview_matrix(layer->W, 4);
        std::cout << "Layer biases:\n";
        preview_vector(layer->b, 4);

        delete layer->Z;
        delete layer->A;

        layer->Z = new Matrix(layer->W->rows, A->cols);
        *layer->Z = matmul(*layer->W, *A);
        std::cout << "After matmul:\n";
        preview_matrix(layer->Z, 4);

        *layer->Z = *layer->Z + *layer->b;
        std::cout << "After adding bias (Z):\n";
        preview_matrix(layer->Z, 4);

        layer->A = new Matrix(layer->Z->rows, layer->Z->cols);
        *layer->A = *layer->Z;  // Copy Z to A

        std::cout << "Applying " << layer->activation << " activation\n";
        if (layer->activation == "Sigmoid") {
            sigmoid.function(*layer->A);
        }
        else if (layer->activation == "Softmax") {
            softmax.function(*layer->A);
        }

        std::cout << "After activation (A):\n";
        preview_matrix(layer->A, 4);

        A = layer->A;
    }

    std::cout << "Finished forward propagation\n";
}

// void NeuralNetwork::forward(Matrix& X) {
//     // Set the input matrix
//     Matrix* A = &X;
// 
//     // Activation function objects
//     Sigmoid sigmoid;
//     Softmax softmax;
// 
//     // Iterate through each layer
//     for (Layer* layer : layers) {
//         // Clean up previous Z and A if they exist
//         if (layer->Z != nullptr) {
//             delete layer->Z;
//         }
//         if (layer->A != nullptr) {
//             delete layer->A;
//         }
// 
//         // Compute Z = W * A + b
//         layer->Z = new Matrix(layer->W->rows, A->cols);
//         *layer->Z = matmul(*layer->W, *A);
//         *layer->Z = *layer->Z + *layer->b;
// 
//         // Compute A = activation(Z)
//         layer->A = new Matrix(layer->Z->rows, layer->Z->cols);
//         *layer->A = *layer->Z;  // Copy Z to A
// 
//         if (layer->activation == "Sigmoid") {
//             sigmoid.function(*layer->A);
//         }
//         else if (layer->activation == "Softmax") {
//             softmax.function(*layer->A);
//         }
// 
//         A = layer->A;
//     }
// }
