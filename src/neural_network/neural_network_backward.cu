/**
 * @file neural_network_backward.cu
 * @brief Implementation of the NeuralNetwork::backward method.
 */

#include "neural_network.h"

void NeuralNetwork::backward(const Matrix& X, const Matrix& Y) {
    // Get the number of training examples
    int m = X.get_cols();

    // Compute the gradient of the output layer
    // DZ2 = A2 - Y
    DZ2 = A2.subtract(Y);

    // Compute gradient for W2
    // DW2 = 1/m * DZ2 * A1^T
    DW2 = DZ2.multiply(A1.transpose());
    DW2.divide_scalar(m);

    // Compute gradient for b2
    // db2 = 1/m * sum(DZ2)
    db2 = DZ2.sum() / m;

    // Compute the gradient of the hidden layer
    // DZ1 = W2^T * DZ2 .* ReLU'(Z1)
    Matrix W2_transpose = W2.transpose();
    DZ1 = W2_transpose.multiply(DZ2);
    Matrix Z1_relu_derivative = Z1.relu_derivative();
    DZ1 = DZ1.multiply_elementwise(Z1_relu_derivative);

    // Compute gradient for W1
    // DW1 = 1/m * DZ1 * X^T
    DW1 = DZ1.multiply(X.transpose());
    DW1.divide_scalar(m);

    // Compute gradient for b1
    // db1 = 1/m * sum(DZ1)
    db1 = DZ1.sum() / m;
}
