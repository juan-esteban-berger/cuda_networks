#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
// Define constants for number of rows and neurons
    const int NUM_ROWS_TRAIN = 1000;
    const int NUM_ROWS_TEST = 10000;
    const int NUM_NEURONS_INPUT = 784;
    const int NUM_NEURONS_HIDDEN_1 = 64;
    const int NUM_NEURONS_OUTPUT = 10;

////////////////////////////////////////////////////////////////////////
// Load, Preprocess, and Preview Data
    // Read in data from X_train.csv
    Matrix X_train;
    initialize_matrix(&X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);
    read_csv("data/X_train.csv", &X_train);
    printf("X_train:\n");
    preview_matrix(&X_train, 2);

    // Read in data from Y_train.csv
    Matrix Y_train;
    initialize_matrix(&Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);
    read_csv("data/Y_train.csv", &Y_train);
    printf("Y_train:\n");
    preview_matrix(&Y_train, 2);

    // Read in data from X_test.csv
    Matrix X_test;
    initialize_matrix(&X_test, NUM_ROWS_TEST, NUM_NEURONS_INPUT);
    read_csv("data/X_test.csv", &X_test);

    // Read in data from Y_test.csv
    Matrix Y_test;
    initialize_matrix(&Y_test, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);
    read_csv("data/Y_test.csv", &Y_test);

    // Preview the first image from X_train
    printf("First image from X_train:\n");
    preview_image(&X_train, 0, 28, 28);

    // Normalize X_train and X_test (after previewing image)
    normalize_matrix(&X_train, 0, 255);
    normalize_matrix(&X_test, 0, 255);

////////////////////////////////////////////////////////////////////////
// Initialize Neural Network
    NeuralNetwork nn;
    initialize_neural_network(&nn,
			      NUM_NEURONS_INPUT,
			      NUM_NEURONS_HIDDEN_1,
	                      NUM_NEURONS_OUTPUT);

////////////////////////////////////////////////////////////////////////
// Train Model
    printf("Training on Training Dataset:\n");
    train(&nn, &X_train, &Y_train, 100, 0.1);
    // train(&nn, &X_train, &Y_train, 1, 0.1);

////////////////////////////////////////////////////////////////////////
// Save Model
    printf("Saving Model:\n");
    save_model("models/nn.csv", &nn);

////////////////////////////////////////////////////////////////////////
// Load Model
    printf("Loading Model:\n");
    NeuralNetwork nn_loaded;
    initialize_neural_network(&nn_loaded,
			      NUM_NEURONS_INPUT,
			      NUM_NEURONS_HIDDEN_1,
	                      NUM_NEURONS_OUTPUT);

    load_model("models/nn.csv", &nn_loaded);

////////////////////////////////////////////////////////////////////////
// Make Predictions
    Matrix Y_pred;
    initialize_matrix(&Y_pred, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);
    printf("Testing on Testing Dataset:\n");
    predict(&nn_loaded, &X_test, &Y_test, &Y_pred);

////////////////////////////////////////////////////////////////////////
// Compare a few predictions
    denormalize_matrix(&X_test, 0, 255);
    printf("Previewing a few predictions:\n");
    preview_predictions(&X_test, &Y_pred, 28, 28, 5);

////////////////////////////////////////////////////////////////////////
// TODO
	// Add one more layer (so it counts as deep learning)
		// Modify Code so it works
	// Modify code to work with both cuda and c
		// Make sure it works with cuda

////////////////////////////////////////////////////////////////////////
// Free memory
    free_matrix(&X_train);
    free_matrix(&Y_train);
    free_matrix(&X_test);
    free_matrix(&Y_test);

    free_neural_network(&nn);
    free_neural_network(&nn_loaded);

    free_matrix(&Y_pred);

    return 0;
}
