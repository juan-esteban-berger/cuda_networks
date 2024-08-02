#include <iostream>
#include <iomanip>
#include <string>

#include "linear_algebra.h"
#include "neural_network.h"

int main() {
//////////////////////////////////////////////////////////////////
// Load Data
    std::cout << "Loading Data..." << std::endl;
    // Define matrix dimensions
    int X_test_rows = 10000;
    int X_test_cols = 784;
    int Y_test_rows = 10000;
    int Y_test_cols = 10;
    
    // Initialize matrices
    Matrix X_test(X_test_rows, X_test_cols);
    Matrix Y_test(Y_test_rows, Y_test_cols);
    
    // Read data
    read_csv("data/X_test.csv", &X_test);
    read_csv("data/Y_test.csv", &Y_test);

    // Print Shape
    std::cout << "X_test: (" 
              << X_test.rows << ", " 
              << X_test.cols << ")" 
              << std::endl;

    std::cout << "Y_test: (" 
              << Y_test.rows << ", " 
              << Y_test.cols << ")" 
              << std::endl;

//////////////////////////////////////////////////////////////////
// Transpose Data
    Matrix* X_test_T = transpose_matrix(&X_test);
    Matrix* Y_test_T = transpose_matrix(&Y_test);

//////////////////////////////////////////////////////////////////
// Normalize X values
    normalize_matrix(X_test_T, 0, 255);

//////////////////////////////////////////////////////////////////
// Function to Display Image

//////////////////////////////////////////////////////////////////
// Test Neural Network
    NeuralNetwork nn;

    std::cout << "Loading Model..." << std::endl;
    nn.load_config("models/cpp_config.csv");
    nn.load_weights("models/cpp_weights.csv");

    std::cout << "Testing..." << std::endl;
    Vector pred = nn.predict(*X_test_T);
    double acc = nn.get_accuracy(*Y_test_T);
    std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << acc << std::endl;

//////////////////////////////////////////////////////////////////
// Preview 5 Random Images

    return 0;
}
