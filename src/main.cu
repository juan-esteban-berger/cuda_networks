#include <iostream>
#include "neural_network/neural_network.h"

int main() {
    try {
        // Set parameters
        int num_examples = 60000;
        int input_size = 784;
        int output_size = 10;
        int hidden_size = 200;  // Changed from 10 to 200
        double learning_rate = 0.001;
        int epochs = 200;  // Changed from 1 to 200

        // Create input and output matrices
        Matrix X_train(num_examples, input_size);
        Matrix Y_train(num_examples, output_size);

        std::cout << "Reading training data..." << std::endl;
        X_train.read_csv("data/X_train.csv");
        Y_train.read_csv("data/Y_train.csv");

        // Transpose X_train and Y_train
        std::cout << "Transposing matrices..." << std::endl;
        Matrix X_train_transposed = X_train.transpose();
        Matrix Y_train_transposed = Y_train.transpose();

        // Create neural network
        std::cout << "Creating neural network..." << std::endl;
        NeuralNetwork nn(input_size, hidden_size, output_size);

        // Perform gradient descent
        std::cout << "Training neural network..." << std::endl;
        nn.gradient_descent(X_train_transposed, Y_train_transposed, learning_rate, epochs);

        // Calculate final accuracy
        double final_accuracy = nn.get_accuracy(Y_train_transposed);
        std::cout << "Final training accuracy: " << final_accuracy << std::endl;

        std::cout << "Training completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
