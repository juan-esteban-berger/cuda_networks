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

// //////////////////////////////////////////////////////////////////
// // Layer Class
// class Layer {
// public:
//     Matrix* W;
//     Vector* b;
//     Matrix* Z;
//     Matrix* A;
//     Matrix* dZ;
//     Matrix* dW;
//     Vector* db;
//     std::string activation;
// 
//     Layer(int input_num, int output_num, std::string activation_func);
//     ~Layer();
// };
// 
// //////////////////////////////////////////////////////////////////
// // Neural Network Class
// class NeuralNetwork {
// public:
//     NeuralNetwork();
//     ~NeuralNetwork();
//     void add_layer(Layer* layer);
//     Matrix* getOutput();
//     void forward(Matrix& X);
// 
// private:
//     std::vector<Layer*> layers;
// };

// def forward(self, X):
//     A = X
//     for layer in self.layers:
//         layer.Z = layer.W.dot(A) + layer.b
//         layer.A = layer.activation.function(layer.Z)
//         A = layer.A

// #ifndef LINEAR_ALGEBRA_H
// #define LINEAR_ALGEBRA_H
// 
// //////////////////////////////////////////////////////////////////
// // Vector Class
// class Vector {
// public:
//     double* data;
//     int rows;
// 
//     Vector(int r);
//     ~Vector();
//     void setValue(int index, double value);
//     double getValues(int index);
// };
// 
// //////////////////////////////////////////////////////////////////
// // Matrix Class
// class Matrix {
// public:
//     double** data;
//     int rows;
//     int cols;
// 
//     Matrix(int r, int c);
//     ~Matrix();
//     void setValue(int row, int col, double value);
//     double getValues(int row, int col);
// };
// 
// //////////////////////////////////////////////////////////////////
// // Matrix and Vector Operations
// // Element-wise multiplication
// Matrix operator*(Matrix& m1, Matrix& m2);
// 
// // Matrix multiplication
// Matrix matmul(Matrix& m1, Matrix& m2);
// 
// // Matrix-vector addition
// Matrix operator+(Matrix& m, Vector& v);
// 
// //////////////////////////////////////////////////////////////////
// // Read from CSV
// void read_csv(const char* filename, Matrix* matrix);
// void read_csv_limited(const char* filename, Matrix* matrix_subset, int startRow, int endRow, int fileRows, int fileCols);
// 
// /////////////////////////////////////////////////////////////////
// // Preview Functions
// void preview_matrix(Matrix* m, int decimals);
// void preview_vector(Vector* v, int decimals);
// 
// //////////////////////////////////////////////////////////////////
// // Randomize Functions
// void random_vector(Vector* v);
// void random_matrix(Matrix* m);
// 
// //////////////////////////////////////////////////////////////////
// // Transpose Function
// Matrix* transpose_matrix(Matrix* m);
// 
// //////////////////////////////////////////////////////////////////
// // Normalization Functions
// void normalize_vector(Vector* v, double min, double max);
// void normalize_matrix(Matrix* m, double min, double max);
// void denormalize_vector(Vector* v, double min, double max);
// void denormalize_matrix(Matrix* m, double min, double max);
// 
// #endif // LINEAR_ALGEBRA_H

void NeuralNetwork::forward(Matrix& X) {
    // Initialize Matrix A
    Matrix A(X.rows, X.cols);

    // Copy from Matrix X into Matrix A
    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            A.setValue(i, j, X.getValues(i, j));
        }
    }

    // Preview Matrix A
    std::cout << "Matrix A" << std::endl;
    // preview_matrix(A, 4);
    preview_matrix(&A, 4);

    // Initialize Sigmoid object
    Sigmoid sigmoid;
    // Initialize Softmax object
    Softmax softmax;
    
    // Loop through each layer
    for (Layer* layer : layers) {
        // Preview Matrix W
        std::cout << "Matrix W" << std::endl;
        preview_matrix(layer->W, 4);

        // Preview Vector b
        std::cout << "Vector b" << std::endl;
        preview_vector(layer->b, 4);

        // Multiply weights by input matrix
        Matrix Z_temp = matmul(*layer->W, A);

        // Preview Matrix Z1_temp
        std::cout << "Matrix Z_temp" << std::endl;
        preview_matrix(&Z_temp, 4);

        // Add bias to Z matrix
        Matrix Z = Z_temp + *layer->b;
        
        // Create a new Matrix for Z and copy values
        layer->Z = new Matrix(Z.rows, Z.cols);
        for (int i = 0; i < Z.rows; i++) {
            for (int j = 0; j < Z.cols; j++) {
                layer->Z->setValue(i, j, Z.getValues(i, j));
            }
        }
        
        // Preview Matrix Z
        std::cout << "Matrix Z" << std::endl;
        preview_matrix(layer->Z, 4);

        // if (layer->activation == "Sigmoid")
        if (layer->activation == "Sigmoid") {
            // Compute Sigmoid function
            sigmoid.function(Z);

            // Preview Matrix Z
            std::cout << "Matrix Z after Sigmoid" << std::endl;
            preview_matrix(&Z, 4);

        } else if (layer->activation == "Softmax") {
            // Compute Softmax function
            softmax.function(Z);

            // Preview Matrix Z
            std::cout << "Matrix Z after Softmax" << std::endl;
            preview_matrix(&Z, 4);

        }

        // Create a new matrix for A and
        // copy values from Z into A
        layer->A = new Matrix(Z.rows, Z.cols);
        for (int i = 0; i < Z.rows; i++) {
            for (int j = 0; j < Z.cols; j++) {
                layer->A->setValue(i, j, Z.getValues(i, j));
            }
        }

        // Preview Matrix A
        std::cout << "Matrix A" << std::endl;
        preview_matrix(layer->A, 4);

        // // Copy values from A into A
        // A = Matrix(layer->Z->rows, layer->Z->cols);
        // for (int i = 0; i < layer->Z->rows; i++) {
        //     for (int j = 0; j < layer->Z->cols; j++) {
        //         A.setValue(i, j, layer->Z->getValues(i, j));
        //     }
        // }

        // Preview Matrix A
        std::cout << "Matrix A" << std::endl;
        preview_matrix(&A, 4);

    }

    std::cout << "Got to end of function" << std::endl;
}
