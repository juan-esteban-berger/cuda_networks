/**
 * @file neural_network_gradient_descent.cu
 * @brief Implementation of the NeuralNetwork::gradient_descent method.
 */
#include "neural_network.h"
#include <iostream>
#include <iomanip>
#include <string>

void NeuralNetwork::gradient_descent(const Matrix& X, const Matrix& Y, double learning_rate, int epochs) {
    const int bar_width = 50;
    std::string bar;

    // Iterate through epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Perform forward propagation
        forward(X);

        // Perform backward propagation
        backward(X, Y);

        // Update parameters
        update_params(learning_rate);

        // Calculate accuracy for this epoch
        double accuracy = get_accuracy(Y);

        // Calculate progress
        float progress = static_cast<float>(epoch + 1) / epochs;
        int pos = static_cast<int>(bar_width * progress);

        // Update progress bar
        bar = "[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) bar += "=";
            else if (i == pos) bar += ">";
            else bar += " ";
        }
        bar += "] ";

        // Print progress bar and accuracy
        std::cout << "\r" << std::setw(3) << static_cast<int>(progress * 100.0) << "% "
                  << bar << std::setw(3) << epoch + 1 << "/" << std::setw(3) << epochs
                  << " - Accuracy: " << std::fixed << std::setprecision(4) << accuracy << std::flush;
    }
    // Move to the next line after completion
    std::cout << std::endl;
}
