#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "cuda_neural_network.h"
// #include "linear_algebra.h"
#include "cuda_linear_algebra.h"

////////////////////////////////////////////////////////////////////////
// Main function
int main() {
////////////////////////////////////////////////////////////////////////
// Setup
    srand(time(NULL));
    printf("\n");

////////////////////////////////////////////////////////////////////////
// Define constants for number of rows and neurons
    const int NUM_ROWS_TRAIN = 60000;
    const int NUM_ROWS_TEST = 250;
    const int NUM_NEURONS_INPUT = 784;
    const int NUM_NEURONS_HIDDEN_1 = 512;
    const int NUM_NEURONS_HIDDEN_2 = 512;
    const int NUM_NEURONS_OUTPUT = 10;

    const int BATCH_SIZE = 1000;

////////////////////////////////////////////////////////////////////////
// Load, Preprocess, and Preview Data
    // Read in data from X_train.csv
    Matrix X_train;
    initialize_matrix(&X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);
    read_csv("data/X_train.csv", &X_train);
    printf("X_train:\n");
    preview_matrix(&X_train, 2);

    // Copy X_train to device
    Matrix_GPU d_X_train;
    initialize_matrix_on_device(&d_X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);
    copy_matrix_to_device(&X_train, &d_X_train);

    // Read in data from Y_train.csv
    Matrix Y_train;
    initialize_matrix(&Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);
    read_csv("data/Y_train.csv", &Y_train);
    printf("Y_train:\n");
    preview_matrix(&Y_train, 2);

    // Copy Y_train to device
    Matrix_GPU d_Y_train;
    initialize_matrix_on_device(&d_Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);
    copy_matrix_to_device(&Y_train, &d_Y_train);

    // Read in data from X_test.csv
    Matrix X_test;
    initialize_matrix(&X_test, NUM_ROWS_TEST, NUM_NEURONS_INPUT);
    read_csv("data/X_test.csv", &X_test);

    // Copy X_test to device
    Matrix_GPU d_X_test;
    initialize_matrix_on_device(&d_X_test, NUM_ROWS_TEST, NUM_NEURONS_INPUT);
    copy_matrix_to_device(&X_test, &d_X_test);

    // Read in data from Y_test.csv
    Matrix Y_test;
    initialize_matrix(&Y_test, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);
    read_csv("data/Y_test.csv", &Y_test);

    // Copy Y_test to device
    Matrix_GPU d_Y_test;
    initialize_matrix_on_device(&d_Y_test, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);
    copy_matrix_to_device(&Y_test, &d_Y_test);

    // Preview the first image from X_train
    printf("First image from X_train:\n");
    preview_image(&X_train, 0, 28, 28);

    // Normalize X_train and X_test (after previewing image)
    normalize_matrix(&X_train, 0, 255);
    normalize_matrix(&X_test, 0, 255);

    // Initialize Y_pred
    Matrix Y_pred;
    initialize_matrix(&Y_pred, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);

    // Initialize Y_pred on device
    Matrix_GPU d_Y_pred;
    initialize_matrix_on_device(&d_Y_pred, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);

////////////////////////////////////////////////////////////////////////
// Initialize Neural Networks
    // // Initialize Neural Network on host
    // NeuralNetwork nn;
    // initialize_neural_network(&nn,
    //     		      NUM_NEURONS_INPUT,
    //     		      NUM_NEURONS_HIDDEN_1,
    //     		      NUM_NEURONS_HIDDEN_2,
    //                           NUM_NEURONS_OUTPUT);

    // // Initialize Neural Network on device
    // NeuralNetwork_GPU d_nn;
    // initialize_neural_network_on_device(&d_nn,
    //     			     NUM_NEURONS_INPUT,
    //     			     NUM_NEURONS_HIDDEN_1,
    //     			     NUM_NEURONS_HIDDEN_2,
    //     			     NUM_NEURONS_OUTPUT);

    // // Copy Neural Network from host to device
    // copy_neural_network_to_device(&nn, &d_nn);

////////////////////////////////////////////////////////////////////////
// Train and Test Model on Host
    // Train Neural Network using CPU
    // printf("Training on Training Dataset:\n");
    // train(&nn, &X_train, &Y_train, 10, 0.1); // For testing

    // Predict using CPU
    // printf("Testing on Testing Dataset:\n");
    // predict(&nn, &X_test, &Y_test, &Y_pred);
    
////////////////////////////////////////////////////////////////////////
// Train Model on Device
    // CUDA kernel configuration
    dim3 threadsPerBlock (16, 16, 1); // A 16 x 16 block threads
    dim3 numBlocks ((BATCH_SIZE / threadsPerBlock.x) + 1,
		    (BATCH_SIZE / threadsPerBlock.y) + 1,
		    1);
    int sharedMemSize = sizeof(float) * 32 * 32;

    // // Read in data in subsets of subset_size rows
    // int subset_size = 1000;
    // Matrix X_train_subset;
    // initialize_matrix(&X_train_subset, subset_size, NUM_NEURONS_INPUT);

    // int epochs_num = 300;
    // for (int i = 0; i < epochs_num; i++) {
    //     for (int i = 0; i < NUM_ROWS_TRAIN; i += subset_size) {
    //         read_csv_limited("data/X_train.csv", &X_train_subset, i, i + subset_size - 1, 60000, 784);
    //         normalize_matrix(&X_train_subset, 0, 255);
    //         copy_matrix_to_device(&X_train_subset, &d_X_train);
    //         train_GPU(&d_nn, &d_X_train, &d_Y_train, 5, 0.01,
    //         				  threadsPerBlock,
    //         				  numBlocks,
    //         				  sharedMemSize);
    //     }
    // }
    NeuralNetwork nn;
    initialize_neural_network(&nn,
			      NUM_NEURONS_INPUT,
			      NUM_NEURONS_HIDDEN_1,
			      NUM_NEURONS_HIDDEN_2,
	                      NUM_NEURONS_OUTPUT);

    load_model("models/nn.csv", &nn);
    // Initialize Neural Network on device
    NeuralNetwork_GPU d_nn;
    initialize_neural_network_on_device(&d_nn,
				     NUM_NEURONS_INPUT,
				     NUM_NEURONS_HIDDEN_1,
				     NUM_NEURONS_HIDDEN_2,
				     NUM_NEURONS_OUTPUT);

    // Copy Neural Network from host to device
    copy_neural_network_to_device(&nn, &d_nn);

    Matrix_GPU d_X_train_subset;
    initialize_matrix_on_device(&d_X_train_subset, BATCH_SIZE, NUM_NEURONS_INPUT);

    Matrix_GPU d_Y_train_subset;
    initialize_matrix_on_device(&d_Y_train_subset, BATCH_SIZE, NUM_NEURONS_OUTPUT);


    int j = 0;
    do {
    printf("Epoch %d\n", j);
    for (int i = 0; i < NUM_ROWS_TRAIN; i += BATCH_SIZE) {
    // printf("Training on subset %d to %d\n", i, i + BATCH_SIZE - 1);
    // sleep(1);
    // Initialize Neural Network on host

    	// copy_matrix_range_to_matrix_GPU(&d_X_train,
    	// 		&d_X_train_subset,
    	// 		i, i + BATCH_SIZE - 1);
    	// copy_matrix_range_to_matrix_GPU(&d_Y_train,
    	// 		&d_Y_train_subset,
    	// 		i, i + BATCH_SIZE - 1);
        copy_random_matrix_range_to_matrix_GPU(&d_X_train, &d_X_train_subset,
					       &d_Y_train, &d_Y_train_subset,
						BATCH_SIZE, NUM_ROWS_TRAIN);
    	train_GPU(&d_nn, &d_X_train_subset, &d_Y_train_subset, 1, 0.001,
    		    threadsPerBlock,
    		    numBlocks,
    		    sharedMemSize);

    }

    j++;
    
    // Sleep for 1 second
    // sleep(1);
    } while (j < 100);

    // Copy Neural Network from device to host
    copy_neural_network_to_host(&d_nn, &nn);


    free_matrix_on_device(&d_X_train_subset);
    free_matrix_on_device(&d_Y_train_subset);

    save_model("models/nn.csv", &nn);

    copy_neural_network_to_device(&nn, &d_nn);
    free_neural_network(&nn);
    free_neural_network_on_device(&d_nn);


    // Train Neural Network using GPU
//     train_GPU(&d_nn, &d_X_train, &d_Y_train, 1, 0.01,
// 		    threadsPerBlock,
// 		    numBlocks,
// 		    sharedMemSize);

    // Predict using GPU
    // printf("Predicting using GPU:\n");
    // predict_GPU(&d_nn, &d_X_test, &d_Y_test, &d_Y_pred,
    // 		threadsPerBlock, numBlocks, sharedMemSize);

    // Copy Neural Network from device to host
    // copy_neural_network_to_host(&d_nn, &nn);

    // Copy d_Y_pred to Y_pred
    // copy_matrix_to_host(&Y_pred, &d_Y_pred);

////////////////////////////////////////////////////////////////////////
// Save Model
    // printf("Saving Model:\n");
    // save_model("models/nn.csv", &nn);

////////////////////////////////////////////////////////////////////////
// Load Model
    printf("Loading Model:\n");
    NeuralNetwork nn_loaded;
    initialize_neural_network(&nn_loaded,
        		      NUM_NEURONS_INPUT,
        		      NUM_NEURONS_HIDDEN_1,
        		      NUM_NEURONS_HIDDEN_2,
                              NUM_NEURONS_OUTPUT);

    load_model("models/nn.csv", &nn_loaded);

////////////////////////////////////////////////////////////////////////
// Compare a few predictions
    // denormalize_matrix(&X_test, 0, 255);
    // Train Neural Network using CPU
    // printf("Training on Training Dataset:\n");
    // train(&nn, &X_train, &Y_train, 10, 0.1); // For testing

    // Predict using CPU
    printf("Testing on Testing Dataset:\n");
    predict(&nn_loaded, &X_test, &Y_test, &Y_pred);
    printf("Previewing a few predictions:\n");
    denormalize_matrix(&X_test, 0, 255);
    preview_predictions(&X_test, &Y_pred, 28, 28, 5);

////////////////////////////////////////////////////////////////////////
// Free memory
    free_matrix(&X_train);
    free_matrix(&Y_train);
    free_matrix(&X_test);
    free_matrix(&Y_test);

    // free_neural_network(&nn);
    // free_neural_network(&nn_loaded);

    // free_neural_network_on_device(&d_nn);

    free_matrix(&Y_pred);
    free_matrix_on_device(&d_Y_pred);

    return 0;
}
