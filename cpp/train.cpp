#include <iostream>
#include <string>

#include "linear_algebra.h"
#include "neural_network.h"

int main() {
//////////////////////////////////////////////////////////////////
// Load Data
    std::cout << "Loading Data..." << std::endl;
    // // Define matrix dimensions
    // int X_train_rows = 60000;
    // int X_train_cols = 784;
    // int Y_train_rows = 60000;
    // int Y_train_cols = 10;
    // 
    // // Initialize matrices
    // Matrix X_train(X_train_rows, X_train_cols);
    // Matrix Y_train(Y_train_rows, Y_train_cols);
    // 
    // // Read data
    // read_csv("data/X_train.csv", &X_train);
    // read_csv("data/Y_train.csv", &Y_train);

    // Define matrix dimensions
    int num_rows = 1000;
    int X_train_rows = num_rows;
    int X_train_cols = 784;
    int Y_train_rows = num_rows;
    int Y_train_cols = 10;

    // Initialize matrices
    Matrix X_train(X_train_rows, X_train_cols);
    Matrix Y_train(Y_train_rows, Y_train_cols);

    // Read data
    read_csv_limited("data/X_train.csv", &X_train,
                     0, X_train_rows, X_train_rows, 784);
    read_csv_limited("data/Y_train.csv", &Y_train,
                     0, Y_train_rows, Y_train_rows, 10);
    
    // Print Shape
    std::cout << "X_train: (" 
              << X_train.rows << ", " 
              << X_train.cols << ")" 
              << std::endl;

    std::cout << "Y_train: (" 
              << Y_train.rows << ", " 
              << Y_train.cols << ")" 
              << std::endl;

//////////////////////////////////////////////////////////////////
// Transpose Data
    Matrix* X_train_T = transpose_matrix(&X_train);
    Matrix* Y_train_T = transpose_matrix(&Y_train);

//////////////////////////////////////////////////////////////////
// Normalize X values
    normalize_matrix(X_train_T, 0, 255);

//////////////////////////////////////////////////////////////////
// Initialize Neural Network
    NeuralNetwork nn;
    nn.add_layer(new Layer(784, 200, "Sigmoid"));
    nn.add_layer(new Layer(200, 200, "Sigmoid"));
    nn.add_layer(new Layer(200, 10, "Softmax"));

    std::cout << "Training..." << std::endl;
    // int epochs = 1000;
    int epochs = 100;
    double learning_rate = 0.1;
    std::string loss = "CatCrossEntropy";
    std::string history_path = "models/cpp_history.csv";
    nn.train(*X_train_T,
             *Y_train_T,
             epochs,
             learning_rate,
             loss,
             history_path);

    std::cout << "Saving Model..." << std::endl;
    nn.save_config("models/cpp_config.csv");
    nn.save_weights("models/cpp_weights.csv");

    return 0;
}
