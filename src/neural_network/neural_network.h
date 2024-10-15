/**
 * @file neural_network.h
 * @brief Defines the NeuralNetwork class for a simple feedforward neural network.
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "../linear_algebra/matrix.h"
#include "../linear_algebra/vector.h"

/**
 * @class NeuralNetwork
 * @brief Represents a simple feedforward neural network with one hidden layer.
 */
class NeuralNetwork {
public:
    /**
     * @brief Construct a new NeuralNetwork object
     * @param input_size Number of input features
     * @param hidden_size Number of neurons in the hidden layer
     * @param output_size Number of output classes
     */
    NeuralNetwork(int input_size, int hidden_size, int output_size);

    /**
     * @brief Destroy the NeuralNetwork object
     */
    ~NeuralNetwork();

    /**
     * @brief Initialize the neural network parameters
     */
    void initialize();

    /**
     * @brief Get the pointer to the W1 matrix data
     * @return Pointer to the W1 matrix data on the device
     */
    double* get_W1_data() const { return W1.get_data(); }

    /**
     * @brief Get the pointer to the W2 matrix data
     * @return Pointer to the W2 matrix data on the device
     */
    double* get_W2_data() const { return W2.get_data(); }

    /**
     * @brief Get the pointer to the b1 vector data
     * @return Pointer to the b1 vector data on the device
     */
    double* get_b1_data() const { return b1.get_data(); }

    /**
     * @brief Get the pointer to the b2 vector data
     * @return Pointer to the b2 vector data on the device
     */
    double* get_b2_data() const { return b2.get_data(); }

    /**
     * @brief Get the dimensions of the W1 matrix
     * @return std::pair<int, int> containing rows and columns of W1
     */
    std::pair<int, int> get_W1_dimensions() const { return {W1.get_rows(), W1.get_cols()}; }

    /**
     * @brief Get the dimensions of the W2 matrix
     * @return std::pair<int, int> containing rows and columns of W2
     */
    std::pair<int, int> get_W2_dimensions() const { return {W2.get_rows(), W2.get_cols()}; }

    /**
     * @brief Get the size of the b1 vector
     * @return Size of the b1 vector
     */
    int get_b1_size() const { return b1.get_rows(); }

    /**
     * @brief Get the size of the b2 vector
     * @return Size of the b2 vector
     */
    int get_b2_size() const { return b2.get_rows(); }

private:
    int input_size;    ///< Number of input features
    int hidden_size;   ///< Number of neurons in the hidden layer
    int output_size;   ///< Number of output classes

    Matrix W1;         ///< Weights for the hidden layer
    Vector b1;         ///< Biases for the hidden layer
    Matrix W2;         ///< Weights for the output layer
    Vector b2;         ///< Biases for the output layer

    Matrix A;          ///< Input matrix
    Matrix Z1;         ///< Pre-activation of hidden layer
    Matrix A1;         ///< Activation of hidden layer
    Matrix Z2;         ///< Pre-activation of output layer
    Matrix A2;         ///< Activation of output layer (final output)

    Matrix DZ2;        ///< Gradient of Z2
    Matrix DW2;        ///< Gradient of W2
    Vector Db2;        ///< Gradient of b2
    Matrix DZ1;        ///< Gradient of Z1
    Matrix DW1;        ///< Gradient of W1
    Vector Db1;        ///< Gradient of b1
};

#endif // NEURAL_NETWORK_H
