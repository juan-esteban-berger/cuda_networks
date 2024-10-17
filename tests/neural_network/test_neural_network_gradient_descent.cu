/**
 * @file test_neural_network_gradient_descent.cu
 * @brief Test file for the NeuralNetwork::gradient_descent method.
 */
#include <gtest/gtest.h>
#include "../../src/neural_network/neural_network.h"
#include <iostream>

TEST(NeuralNetworkGradientDescentTest, TrainOnMNIST) {
    try {
        // Set parameters
        int num_examples = 60000;
        int input_size = 784;
        int output_size = 10;
        int hidden_size = 10;
        double learning_rate = 0.001;
        int epochs = 1;

        // Create input and output matrices
        Matrix X_train(num_examples, input_size);  // Changed dimensions
        Matrix Y_train(num_examples, output_size);  // Changed dimensions

        X_train.read_csv("data/X_train.csv");
        Y_train.read_csv("data/Y_train.csv");

        // Print matrices after reading
        std::cout << "X_train:" << std::endl;
        // X_train.print(2);
        
        std::cout << "Y_train:" << std::endl;
        // Y_train.print(2);

        // Transpose X_train and Y_train
        Matrix X_train_transposed = X_train.transpose();
        Matrix Y_train_transposed = Y_train.transpose();

        // Print matrices after transposing
        std::cout << "X_train_transposed:" << std::endl;
        // X_train_transposed.print(2);
        
        std::cout << "Y_train_transposed:" << std::endl;
        // Y_train_transposed.print(2);

        // Create neural network
        NeuralNetwork nn(input_size, hidden_size, output_size);

        // Perform gradient descent
        nn.gradient_descent(X_train_transposed, Y_train_transposed, learning_rate, epochs);

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        FAIL() << "Exception thrown during test execution";
    }
}
