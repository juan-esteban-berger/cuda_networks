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

// Neural Network Struct for GPU
typedef struct {
    Matrix_GPU W1;
    Vector b1;
    Matrix_GPU W2;
    Vector b2;
    Matrix_GPU WOutput;
    Vector bOutput;
} NeuralNetwork_GPU;

////////////////////////////////////////////////////////////////////////
// Allocate and free memory
void initialize_neural_network(NeuralNetwork* nn,
		int input_neurons,
		int h1_neurons,
		int h2_neurons,
		int output_neurons);

void initialize_neural_network_on_device(NeuralNetwork_GPU* d_nn,
		int input_neurons,
		int h1_neurons,
		int h2_neurons,
		int output_neurons);

void free_neural_network(NeuralNetwork* nn);

void free_neural_network_on_device(NeuralNetwork_GPU* d_nn);

////////////////////////////////////////////////////////////////////////
// Save and Load Models Function
void save_model(const char* filename, NeuralNetwork* nn);
void load_model(const char* filename, NeuralNetwork* nn);

////////////////////////////////////////////////////////////////////////
// Copy Neural Network to and from the host and device
void copy_neural_network_to_device(NeuralNetwork* h_nn,
				   NeuralNetwork_GPU* d_nn);
void copy_neural_network_to_host(NeuralNetwork_GPU* d_nn,
				 NeuralNetwork* h_nn);

////////////////////////////////////////////////////////////////////////
// Activation functions
void ReLU(Matrix* m, Matrix* a);
__global__ void ReLU_GPU(double* input, double* output, int rows, int cols);
void ReLU_derivative(Matrix* m, Matrix* a);
__global__ void ReLU_derivative_GPU(double* input, double* output, int rows, int cols);
void softmax(Matrix* m, Matrix* a);
__global__ void softmax_GPU(double* input, double* output, int rows, int cols);

////////////////////////////////////////////////////////////////////////
// Propagation functions
void forward_propagation(Matrix* X_T,
		Matrix* W1, Vector* b1,
		Matrix* W2, Vector* b2,
		Matrix* WOutput, Vector* bOutput,
		Matrix* Z1, Matrix* A1,
		Matrix* Z2, Matrix* A2,
		Matrix* ZOutput, Matrix* AOutput);

void forward_propagation_GPU(Matrix_GPU* X_T,
		Matrix_GPU* W1, Vector* b1,
		Matrix_GPU* W2, Vector* b2,
		Matrix_GPU* WOutput, Vector* bOutput,
		Matrix_GPU* Z1, Matrix_GPU* A1,
		Matrix_GPU* Z2, Matrix_GPU* A2,
		Matrix_GPU* ZOutput, Matrix_GPU* AOutput,
		dim3 threadsPerBlock,
		dim3 numBlocks,
		int sharedMemSize);

void backward_propagation(Matrix* X_T, Matrix* Y_T,
		Matrix* W1, Vector* b1,
		Matrix* W2, Vector* b2,
		Matrix* WOutput, Vector* bOutput,
		Matrix* Z1, Matrix* A1,
		Matrix* Z2, Matrix* A2,
		Matrix* ZOutput, Matrix* AOutput,
		Matrix* dW1, double* db1,
		Matrix* dW2, double* db2,
		Matrix* dWOutput, double* dbOutput,
		Matrix* dZ1, Matrix* dZ2, Matrix* dZOutput,
		Matrix* WOutput_T,
		Matrix* WOutput_dZOutput,
		Matrix* W2_T,
		Matrix* W2_dZ2,
		Matrix* A2_T, Matrix* A1_T, Matrix* X);

void backward_propagation_GPU(Matrix_GPU* X_T, Matrix_GPU* Y_T,
		              Matrix_GPU* W1, Vector* b1,
		              Matrix_GPU* W2, Vector* b2,
		              Matrix_GPU* WOutput, Vector* bOutput,
		              Matrix_GPU* Z1, Matrix_GPU* A1,
		              Matrix_GPU* Z2, Matrix_GPU* A2,
		              Matrix_GPU* ZOutput, Matrix_GPU* AOutput,
		              Matrix_GPU* dW1, double** db1,
		              Matrix_GPU* dW2, double** db2,
		              Matrix_GPU* dWOutput, double** dbOutput,
		              Matrix_GPU* dZ1, Matrix_GPU* dZ2, Matrix_GPU* dZOutput,
		              Matrix_GPU* WOutput_T,
		              Matrix_GPU* WOutput_dZOutput,
		              Matrix_GPU* W2_T,
		              Matrix_GPU* W2_dZ2,
		              Matrix_GPU* A2_T, Matrix_GPU* A1_T, Matrix_GPU* X,
		              dim3 threadsPerBlock,
		              dim3 numBlocks,
		              int sharedMemSize);

////////////////////////////////////////////////////////////////////////
// Parameter update function
void update_parameters(Matrix* W1, Vector* b1,
		Matrix* W2, Vector* b2,
		Matrix* WOutput, Vector* bOutput,
		Matrix* dW1, double db1,
		Matrix* dW2, double db2,
		Matrix* dWOutput, double dbOutput,
		double learning_rate);

__global__ void update_parameters_GPU(double* W1, double* b1,
				      double* W2, double* b2,
				      double* WOutput, double* bOutput,
				      double* dW1, double* db1,
				      double* dW2, double* db2,
				      double* dWOutput, double* dbOutput,
				      int W1_rows, int W1_cols,
				      int b1_rows,
				      int W2_rows, int W2_cols,
				      int b2_rows,
				      int WOutput_rows, int WOutput_cols,
				      int bOutput_rows,
				      double learning_rate);

////////////////////////////////////////////////////////////////////////
// Accuracy calculation function
void calculate_accuracy(Vector* Y, Vector* Y_hat);
__global__ void calculate_accuracy_GPU(double* Y,
				       double* Y_hat,
				       int rows,
				       double* accuracy);

////////////////////////////////////////////////////////////////////////
// Training and prediction functions
void train(NeuralNetwork* nn, Matrix* X, Matrix* Y, int epochs, double learning_rate);
void gradient_descent_GPU(NeuralNetwork_GPU* d_nn, Matrix_GPU* d_X, Matrix_GPU* d_Y,
	       double learning_rate, double* accuracy,
	       dim3 threadsPerBlock,
	       dim3 numBlocks,
	       int sharedMemSize);
void train_GPU(NeuralNetwork_GPU* d_nn, Matrix_GPU* d_X, Matrix_GPU* d_Y,
	       int epochs, double learning_rate, int batch_size,
	       dim3 threadsPerBlock,
	       dim3 numBlocks,
	       int sharedMemSize);
void predict(NeuralNetwork* nn, Matrix* X, Matrix* Y, Matrix* Y_pred);
void predict_GPU(NeuralNetwork_GPU* d_nn, Matrix_GPU* d_X, Matrix_GPU* d_Y, Matrix_GPU* d_Y_pred,
		 dim3 threadsPerBlock,
		 dim3 numBlocks,
		 int sharedMemSize);

////////////////////////////////////////////////////////////////////////
// Functions for comparing actual and predicted values
void preview_predictions(Matrix* X, Matrix* Y_pred, int image_size_x, int image_size_y, int n);

#endif // NEURAL_NETWORK_H
