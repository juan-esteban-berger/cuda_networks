#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <ctime>

#include "linear_algebra.h"
#include "neural_network.h"

//////////////////////////////////////////////////////////////////
// Function to Display Image
void display_image(Matrix* X, int index) {
    int image_size = 28;
    for (int i = 0; i < image_size; ++i) {
        for (int j = 0; j < image_size; ++j) {
            int linear_index = i * image_size + j;
            double pixel = X->getValues(linear_index, index);
            if (pixel == 0) {
                std::cout << "  ";
            } else {
                std::cout << "##";
            }
        }
        std::cout << std::endl;
    }
}

//////////////////////////////////////////////////////////////////
// Main Function
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
    // Initialize random number generator
    std::mt19937 rng(std::time(0));
    std::uniform_int_distribution<int> dist(0, X_test_T->cols - 1);
    
    std::cout << "Displaying 5 Random Images..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        // Get random index
        int random_index = dist(rng);
        // Get predicted value
        int pred_val = static_cast<int>(pred.getValues(random_index));
        // Initialize true value
        int y_val = 0;
        // Get values from Y_test_T
        double max_prob = Y_test_T->getValues(0, random_index);
        // Loop over rows
        for (int j = 1; j < Y_test_T->rows; ++j) {
            // If value is greater than max_prob
            if (Y_test_T->getValues(j, random_index) > max_prob) {
                // Update max_prob
                max_prob = Y_test_T->getValues(j, random_index);
                // Update y_val
                y_val = j;
            }
        }

        std::cout << "Predicted: " << pred_val;
        std::cout << ", True: " << y_val << std::endl;
        display_image(X_test_T, random_index);

        std::cout << "Press Enter to see the next image..." << std::endl;
        std::cin.get();
    }

    return 0;
}
