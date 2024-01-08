#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "linear_algebra.h"

////////////////////////////////////////////////////////////////////////
// Neural Network Struct
typedef struct {
    Matrix W1;
    Vector b1;
    Matrix W2;
    Vector b2;
    Matrix WOutput;
    Vector bOutput;
} NeuralNetwork;

////////////////////////////////////////////////////////////////////////
// Allocate and free memory
void initialize_neural_network(NeuralNetwork* nn,
		int input_neurons,
		int h1_neurons,
		int h2_neurons,
		int output_neurons);
void free_neural_network(NeuralNetwork* nn);

////////////////////////////////////////////////////////////////////////
// Save and Load Models Function
void save_model(const char* filename, NeuralNetwork* nn);
void load_model(const char* filename, NeuralNetwork* nn);

////////////////////////////////////////////////////////////////////////
// Activation functions
void ReLU(Matrix* m, Matrix* a);
void ReLU_derivative(Matrix* m, Matrix* a);
void softmax(Matrix* m, Matrix* a);

////////////////////////////////////////////////////////////////////////
// Propagation functions
void forward_propagation(Matrix* X_T,
		Matrix* W1, Vector* b1,
		Matrix* W2, Vector* b2,
		Matrix* WOutput, Vector* bOutput,
		Matrix* Z1, Matrix* A1,
		Matrix* Z2, Matrix* A2,
		Matrix* ZOutput, Matrix* AOutput);

void backward_propagation(Matrix* X_T, Matrix* Y_T,
		Matrix* W1, Vector* b1,
		Matrix* WOutput, Vector* bOutput,
		Matrix* Z1, Matrix* Z1_deac, Matrix* A1,
		Matrix* ZOutput, Matrix* AOutput,
		Matrix* dW1, float* db1,
		Matrix* dWOutput, float* dbOutput,
		Matrix* dZ1, Matrix* dZOutput,
		Matrix* WOutput_T, Matrix* WOutput_dZOutput,
		Matrix* A1_T, Matrix* X);

////////////////////////////////////////////////////////////////////////
// Parameter update function
void update_parameters(Matrix* W1, Vector* b1,
		Matrix* W2, Vector* b2,
		Matrix* WOutput, Vector* bOutput,
		Matrix* dW1, float db1,
		Matrix* dW2, float db2,
		Matrix* dWOutput, float dbOutput,
		float learning_rate);

////////////////////////////////////////////////////////////////////////
// Accuracy calculation function
void calculate_accuracy(Vector* Y, Vector* Y_hat);

////////////////////////////////////////////////////////////////////////
// Training and prediction functions
void train(NeuralNetwork* nn, Matrix* X, Matrix* Y, int epochs, float learning_rate);
void predict(NeuralNetwork* nn, Matrix* X, Matrix* Y, Matrix* Y_pred);

////////////////////////////////////////////////////////////////////////
// Functions for comparing actual and predicted values
void preview_predictions(Matrix* X, Matrix* Y_pred, int image_size_x, int image_size_y, int n);

#endif // NEURAL_NETWORK_H
