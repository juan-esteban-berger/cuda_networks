#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "neural_network.h"
#include "linear_algebra.h"

////////////////////////////////////////////////////////////////////////
// Main function
int main() {
////////////////////////////////////////////////////////////////////////
// Setup
    srand(time(NULL));
    printf("\n");

////////////////////////////////////////////////////////////////////////
// Define model location
    char model_name[100] = "models/nn.csv";

////////////////////////////////////////////////////////////////////////
// Define constants for number of rows and neurons
    const int NUM_ROWS_TEST = 2000;
    const int NUM_NEURONS_INPUT = 784;
    const int NUM_NEURONS_HIDDEN_1 = 512;
    const int NUM_NEURONS_HIDDEN_2 = 512;
    const int NUM_NEURONS_OUTPUT = 10;

    const int BATCH_SIZE = 2000;

////////////////////////////////////////////////////////////////////////
// Define CUDA kernel configuration
    dim3 threadsPerBlock (32, 32, 1);
    dim3 numBlocks ((BATCH_SIZE / threadsPerBlock.x) + 1,
		    (BATCH_SIZE / threadsPerBlock.y) + 1,
		    1);
    int sharedMemSize = sizeof(double) * 32 * 32;

////////////////////////////////////////////////////////////////////////
// Read in Testing Data
    // Read in data from X_test.csv
    Matrix X_test;
    initialize_matrix(&X_test, NUM_ROWS_TEST, NUM_NEURONS_INPUT);
    read_csv("data/X_test.csv", &X_test);

    // Read in data from Y_test.csv
    Matrix Y_test;
    initialize_matrix(&Y_test, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);
    read_csv("data/Y_test.csv", &Y_test);

////////////////////////////////////////////////////////////////////////
// Copy Testing Data to Device
    // Copy X_test to device
    Matrix_GPU d_X_test;
    initialize_matrix_on_device(&d_X_test, NUM_ROWS_TEST, NUM_NEURONS_INPUT);
    copy_matrix_to_device(&X_test, &d_X_test);

    // Copy Y_test to device
    Matrix_GPU d_Y_test;
    initialize_matrix_on_device(&d_Y_test, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);
    copy_matrix_to_device(&Y_test, &d_Y_test);

////////////////////////////////////////////////////////////////////////
// Initialize Neural Network on Host
    NeuralNetwork nn;
    initialize_neural_network(&nn,
			      NUM_NEURONS_INPUT,
			      NUM_NEURONS_HIDDEN_1,
			      NUM_NEURONS_HIDDEN_2,
	                      NUM_NEURONS_OUTPUT);

////////////////////////////////////////////////////////////////////////
// Load Model
    printf("Loading Model from %s\n", model_name);
    load_model(model_name, &nn);

////////////////////////////////////////////////////////////////////////
// Initialize Neural Network on Device
    NeuralNetwork_GPU d_nn;
    initialize_neural_network_on_device(&d_nn,
				     NUM_NEURONS_INPUT,
				     NUM_NEURONS_HIDDEN_1,
				     NUM_NEURONS_HIDDEN_2,
				     NUM_NEURONS_OUTPUT);

    // Copy Neural Network from host to device
    copy_neural_network_to_device(&nn, &d_nn);

////////////////////////////////////////////////////////////////////////ZZ
// Make Predictions
    // Initialize Y_pred on device
    Matrix_GPU d_Y_pred;
    initialize_matrix_on_device(&d_Y_pred, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);

    // Make Predictions
    printf("Predicting using GPU:\n");
    predict_GPU(&d_nn, &d_X_test, &d_Y_test, &d_Y_pred,
    		threadsPerBlock, numBlocks, sharedMemSize);

////////////////////////////////////////////////////////////////////////
// Copy Predictions to Host
    // Copy Predictions to host
    Matrix Y_pred;
    initialize_matrix(&Y_pred, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);
    copy_matrix_to_host(&Y_pred, &d_Y_pred);

    // Preview Predictions
    preview_predictions(&X_test, &Y_pred, 28, 28, 5);

////////////////////////////////////////////////////////////////////////
// Free memory
    // Free training and testing data on host and device
    free_matrix(&X_test);
    free_matrix(&Y_test);
    free_matrix_on_device(&d_X_test);
    free_matrix_on_device(&d_Y_test);

    // Free neural networks on host and device
    free_neural_network(&nn);
    free_neural_network_on_device(&d_nn);

    // Free Predictions on device
    free_matrix_on_device(&d_Y_pred);
    free_matrix(&Y_pred);

    return 0;
}
