#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#include "cuda_linear_algebra.h"
#include "cuda_neural_network.h"

////////////////////////////////////////////////////////////////////////
// Main function
int main() {
////////////////////////////////////////////////////////////////////////
// Setup
    srand(time(NULL));
    printf("\n");

    dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
    int N = 64;
    dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

////////////////////////////////////////////////////////////////////////
// Define constants for number of rows and neurons
    // int epochs_num = 300;
    int epochs_num = 1;
    const int NUM_ROWS_TRAIN = 1000;
    // const int NUM_ROWS_TEST = 10000;
    const int NUM_NEURONS_INPUT = 784;
    const int NUM_NEURONS_HIDDEN_1 = 200;
    const int NUM_NEURONS_HIDDEN_2 = 200;
    const int NUM_NEURONS_OUTPUT = 10;

////////////////////////////////////////////////////////////////////////
// Load, Preprocess, and Preview Data
    // Read in data from X_train.csv
    Matrix X_train;
    initialize_matrix(&X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);
    read_csv("data/X_train.csv", &X_train);
    printf("X_train:\n");
    preview_matrix(&X_train, 2);

    // Initialize Matrix d_X_train on device
    printf("Initializing Matrix d_X_train on device:\n");
    Matrix_GPU d_X_train;
    initialize_matrix_on_device(&d_X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);

    // Copy Matrix X_train to device
    printf("Copying Matrix X_train to device:\n");
    copy_matrix_to_device(&X_train, &d_X_train);
    preview_matrix_GPU(&d_X_train, 2);

    // Read in data from Y_train.csv
    Matrix Y_train;
    initialize_matrix(&Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);
    read_csv("data/Y_train.csv", &Y_train);
    printf("Y_train:\n");
    preview_matrix(&Y_train, 2);

    // Initialize Matrix d_Y_train on device
    printf("Initializing Matrix d_Y_train on device:\n");
    Matrix_GPU d_Y_train;
    initialize_matrix_on_device(&d_Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);

    // Copy Matrix Y_train to device
    printf("Copying Matrix Y_train to device:\n");
    copy_matrix_to_device(&Y_train, &d_Y_train);
    preview_matrix_GPU(&d_Y_train, 2);

////////////////////////////////////////////////////////////////////////
// Test CPU Forward Propagation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    // Initialize Neural Network
    printf("Initializing Neural Network:\n");
    NeuralNetwork nn_value;
    initialize_neural_network(&nn_value,
		    	      NUM_NEURONS_INPUT,
			      NUM_NEURONS_HIDDEN_1,
			      NUM_NEURONS_HIDDEN_2,
			      NUM_NEURONS_OUTPUT);

    // Create pointer to Neural Network
    NeuralNetwork *nn = &nn_value;

    // Transpose X to get the correct dimensions for matrix multiplication
    Matrix X_T;
    initialize_matrix(&X_T, X_train.cols, X_train.rows);
    transpose_matrix(&X_train, &X_T);

    // Transpose Y_T to match AOutput
    Matrix Y_T;
    initialize_matrix(&Y_T, Y_train.cols, Y_train.rows);
    transpose_matrix(&Y_train, &Y_T);

    // First Layer
    Matrix Z1;
    initialize_matrix(&Z1, nn->W1.rows, X_T.cols);
    Matrix A1;
    initialize_matrix(&A1, nn->W1.rows, X_T.cols);

    // Second Layer
    Matrix Z2;
    initialize_matrix(&Z2, nn->W2.rows, X_T.cols);
    Matrix A2;
    initialize_matrix(&A2, nn->W2.rows, X_T.cols);

    // Output Layer
    Matrix ZOutput;
    initialize_matrix(&ZOutput, nn->WOutput.rows, X_T.cols);
    Matrix AOutput;
    initialize_matrix(&AOutput, nn->WOutput.rows, X_T.cols);

    // Initialize Vectors for Y and Y_hat
    Vector Y_true;
    initialize_vector(&Y_true, X_T.cols);
    Vector Y_hat;
    initialize_vector(&Y_hat, X_T.cols);

////////////////////////////////////////////////////////////////////////
// Test CPU Backpropagation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CPU Backpropagation Function:\n");

    // Normalize X_train
    printf("Normalizing X_train:\n");
    normalize_matrix(&X_train, 0, 255);
    preview_matrix(&X_train, 2);

    // Preview first image in X_train
    printf("Previewing first image in X_train:\n");
    preview_image(&X_train, 0, 28, 28);

    
    // Vectors/Matrices Needed for Calculation of Output Layer Gradients
    // dZOutput = AOutput - Y_T
    Matrix dZOutput;
    initialize_matrix(&dZOutput, ZOutput.rows, ZOutput.cols);
    // dWOutput = 1/m * matmul(dZOutput, A2_T)
    Matrix dWOutput;
    initialize_matrix(&dWOutput, nn->WOutput.rows, nn->WOutput.cols);
    Matrix A2_T;
    initialize_matrix(&A2_T, A2.cols, A2.rows);
    // dbOutput = 1/m * sum(dZOutput)
    float dbOutput;

    // Vectors/Matrices Needed for Calculation of Second Layer Gradients
    // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_derivative(Z2)
    Matrix dZ2;
    initialize_matrix(&dZ2, Z2.rows, Z2.cols);
    Matrix WOutput_T;
    initialize_matrix(&WOutput_T, nn->WOutput.cols, nn->WOutput.rows);
    Matrix WOutput_dZOutput; // Product of WOutput_T and dZOutput
    initialize_matrix(&WOutput_dZOutput, WOutput_T.rows, dZOutput.cols);
    // dW2 = 1/m * matmul(dZ2, A1_T)
    Matrix dW2;
    initialize_matrix(&dW2, nn->W2.rows, nn->W2.cols);
    Matrix A1_T;
    initialize_matrix(&A1_T, A1.cols, A1.rows);
    // db2 = 1/m * sum(dZ2)
    float db2;

    // Vectors/Matrices Needed for Calculation of First Layer Gradients
    // dZ1 = matmul(W2_T, dZ2) * ReLU_deriv(Z1)
    Matrix dZ1;
    initialize_matrix(&dZ1, Z1.rows, Z1.cols);
    Matrix W2_T;
    initialize_matrix(&W2_T, nn->W2.cols, nn->W2.rows);
    Matrix W2_dZ2; // Product of W2_T and dZ2
    initialize_matrix(&W2_dZ2, W2_T.rows, dZ2.cols);
    // dW1 = 1/m * matmul(dZ1, X_T)
    Matrix dW1;
    initialize_matrix(&dW1, nn->W1.rows, nn->W1.cols);
    // db1 = 1/m * sum(dZ1)
    float db1;

    // Training Rounds
    // int epochs = epochs_num;
    int epochs = 0;
    float learning_rate = 0.1;

    // Loop over the epochs
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d:\n", epoch);

        // Forward Propagation
        forward_propagation(&X_T,
        		&(nn->W1), &(nn->b1),
        	     	&(nn->W2), &(nn->b2),
        		&(nn->WOutput), &(nn->bOutput),
        		&Z1, &A1,
        	     	&Z2, &A2,
        	        &ZOutput, &AOutput);

        // Backward Propagation
        backward_propagation(&X_T, &Y_T,
        		     &(nn->W1), &(nn->b1),
        	      	     &(nn->W2), &(nn->b2),
        		     &(nn->WOutput), &(nn->bOutput),
        		     &Z1, &A1,
        	      	     &Z2, &A2,
        		     &ZOutput, &AOutput,
        		     &dW1, &db1,
        	      	     &dW2, &db2,
        		     &dWOutput, &dbOutput,
        		     &dZ1, &dZ2, &dZOutput,
        	      	     &WOutput_T,
        	             &WOutput_dZOutput,
        	      	     &W2_T,
        	      	     &W2_dZ2,
        		     &A2_T, &A1_T, &X_train);

        // Update Parameters
        update_parameters(&(nn->W1), &(nn->b1),
        	   	  &(nn->W2), &(nn->b2),
        	          &(nn->WOutput), &(nn->bOutput),
        	          &dW1, db1,
        	          &dW2, db2,
        	          &dWOutput, dbOutput,
        	          learning_rate);
        // Get Predictions
        argmax(&Y_T, &Y_true);
        argmax(&AOutput, &Y_hat);

        // Calculate Accuracy
        calculate_accuracy(&Y_true, &Y_hat);
    }

////////////////////////////////////////////////////////////////////////
// Test GPU Forward Propagation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing GPU Forward Propagation Function:\n");

    // Initialize Neural Network on device
    printf("Initializing Neural Network on device:\n");
    NeuralNetwork_GPU d_nn_value;
    initialize_neural_network_on_device(&d_nn_value,
		    		       NUM_NEURONS_INPUT,
				       NUM_NEURONS_HIDDEN_1,
				       NUM_NEURONS_HIDDEN_2,
				       NUM_NEURONS_OUTPUT);

    // Create pointer to Neural Network on device
    NeuralNetwork_GPU *d_nn = &d_nn_value;

    // Define number of threads per block, number of blocks, and shared memory size
    dim3 threads_per_block_fp (16, 16, 1); // A 16 x 16 block threads
    int N_fp = NUM_ROWS_TRAIN;
    dim3 number_of_blocks_fp ((N_fp / threads_per_block_fp.x) + 1, (N_fp / threads_per_block_fp.y) + 1, 1);
    int sharedMemSize_fp = sizeof(float) * 32 * 32;


    // Copy Neural Network to device
    printf("Copying Neural Network to device:\n");
    copy_neural_network_to_device(&nn_value, &d_nn_value);

    // Initialize Matrix d_X_T on device
    printf("Initializing Matrix d_X_T on device:\n");
    Matrix_GPU d_X_T;
    initialize_matrix_on_device(&d_X_T, X_train.cols, X_train.rows);

    // Transpose Matrix d_X_train
    printf("Transposing Matrix d_X_train on device:\n");
    transpose_matrix_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_X_train.data,
		                                                d_X_T.data,
								d_X_train.rows,
								d_X_train.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix d_Y_T on device
    printf("Initializing Matrix d_Y_T on device:\n");
    Matrix_GPU d_Y_T;
    initialize_matrix_on_device(&d_Y_T, Y_train.cols, Y_train.rows);

    // Transpose Matrix d_Y_train
    printf("Transposing Matrix d_Y_train on device:\n");
    transpose_matrix_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_Y_train.data,
		                                                d_Y_T.data,
								d_Y_train.rows,
								d_Y_train.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // First Layer
    Matrix_GPU d_Z1;
    initialize_matrix_on_device(&d_Z1, d_nn->W1.rows, d_X_T.cols);
    Matrix_GPU d_A1;
    initialize_matrix_on_device(&d_A1, d_nn->W1.rows, d_X_T.cols);

    // Second Layer
    Matrix_GPU d_Z2;
    initialize_matrix_on_device(&d_Z2, d_nn->W2.rows, d_X_T.cols);
    Matrix_GPU d_A2;
    initialize_matrix_on_device(&d_A2, d_nn->W2.rows, d_X_T.cols);

    // Output Layer
    Matrix_GPU d_ZOutput;
    initialize_matrix_on_device(&d_ZOutput, d_nn->WOutput.rows, d_X_T.cols);
    Matrix_GPU d_AOutput;
    initialize_matrix_on_device(&d_AOutput, d_nn->WOutput.rows, d_X_T.cols);

    // Initialize Vectors for Y and Y_hat
    Vector d_Y_true;
    initialize_vector_on_device(&d_Y_true, d_X_T.cols);
    Vector d_Y_hat;
    initialize_vector_on_device(&d_Y_hat, d_X_T.cols);

////////////////////////////////////////////////////////////////////////
// Testing GPU Backpropagation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing GPU Backpropagation Function:\n");

    // Copy Normalized X_train to device
    printf("Copying Normalized X_train to device:\n");
    copy_matrix_to_device(&X_train, &d_X_train);
    // preview_matrix_GPU(&d_X_train, 2);

    // Vectors/Matrices Needed for Calculation of Output Layer Gradients
    // dZOutput = AOutput - Y_T
    Matrix_GPU d_dZOutput;
    initialize_matrix_on_device(&d_dZOutput, dZOutput.rows, dZOutput.cols);
    // dWOutput = 1/m * matmul(dZOutput, A2_T)
    Matrix_GPU d_dWOutput;
    initialize_matrix_on_device(&d_dWOutput, nn->WOutput.rows, nn->WOutput.cols);
    Matrix_GPU d_A2_T;
    initialize_matrix_on_device(&d_A2_T, A2.cols, A2.rows);
    // dbOutput = 1/m * sum(dZOutput)
    float* d_dbOutput;
    cudaMalloc((void **)&d_dbOutput, sizeof(float));
    cudaMemset(d_dbOutput, 0, sizeof(float));


    // Vectors/Matrices Needed for Calculation of Second Layer Gradients
    // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_derivative(Z2)
    Matrix_GPU d_dZ2;
    initialize_matrix_on_device(&d_dZ2, dZ2.rows, dZ2.cols);
    Matrix_GPU d_WOutput_T;
    initialize_matrix_on_device(&d_WOutput_T, nn->WOutput.cols, nn->WOutput.rows);
    Matrix_GPU d_WOutput_dZOutput; // Product of WOutput_T and dZOutput
    initialize_matrix_on_device(&d_WOutput_dZOutput, d_WOutput_T.rows, d_dZOutput.cols);
    // dW2 = 1/m * matmul(dZ2, A1_T)
    Matrix_GPU d_dW2;
    initialize_matrix_on_device(&d_dW2, nn->W2.rows, nn->W2.cols);
    copy_matrix_to_device(&dW2, &d_dW2);
    Matrix_GPU d_A1_T;
    initialize_matrix_on_device(&d_A1_T, A1.cols, A1.rows);
    // db2 = 1/m * sum(dZ2)
    float* d_db2;
    cudaMalloc((void **)&d_db2, sizeof(float));
    cudaMemset(d_db2, 0, sizeof(float));

    // Vectors/Matrices Needed for Calculation of First Layer Gradients
    // dZ1 = matmul(W2_T, dZ2) * ReLU_deriv(Z1)
    Matrix_GPU d_dZ1;
    initialize_matrix_on_device(&d_dZ1, dZ1.rows, dZ1.cols);
    copy_matrix_to_device(&dZ1, &d_dZ1);
    Matrix_GPU d_W2_T;
    initialize_matrix_on_device(&d_W2_T, nn->W2.cols, nn->W2.rows);
    copy_matrix_to_device(&W2_T, &d_W2_T);
    Matrix_GPU d_W2_dZ2; // Product of W2_T and dZ2
    initialize_matrix_on_device(&d_W2_dZ2, d_W2_T.rows, d_dZ2.cols);
    copy_matrix_to_device(&W2_dZ2, &d_W2_dZ2);
    // dW1 = 1/m * matmul(dZ1, X_T)
    Matrix_GPU d_dW1;
    initialize_matrix_on_device(&d_dW1, nn->W1.rows, nn->W1.cols);
    copy_matrix_to_device(&dW1, &d_dW1);
    // db1 = 1/m * sum(dZ1)
    float* d_db1;
    cudaMalloc((void **)&d_db1, sizeof(float));
    cudaMemset(d_db1, 0, sizeof(float));

//////////////////////////////////////////////////////////////////
    // Training Rounds
    int d_epochs = epochs_num;
    float d_learning_rate = 0.01;
    for (int epoch = 0; epoch < d_epochs; epoch++) {
        printf("Epoch %d:\n", epoch);

        // Forward Propagation
        forward_propagation_GPU(&d_X_T,
        	    	       &d_nn->W1, &d_nn->b1,
        		       &d_nn->W2, &d_nn->b2,
        		       &d_nn->WOutput, &d_nn->bOutput,
        		       &d_Z1, &d_A1,
        		       &d_Z2, &d_A2,
        		       &d_ZOutput, &d_AOutput,
        		       threads_per_block_fp,
        		       number_of_blocks_fp,
        		       sharedMemSize_fp);

        cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

        // Backward Propagation
        backward_propagation_GPU(&d_X_T, &d_Y_T,
        			 &d_nn->W1, &d_nn->b1,
        	      	         &d_nn->W2, &d_nn->b2,
        		         &d_nn->WOutput, &d_nn->bOutput,
        		         &d_Z1, &d_A1,
        	      	         &d_Z2, &d_A2,
        		         &d_ZOutput, &d_AOutput,
        		         &d_dW1, &d_db1,
        	      	         &d_dW2, &d_db2,
        		         &d_dWOutput, &d_dbOutput,
        		         &d_dZ1, &d_dZ2, &d_dZOutput,
        	      	         &d_WOutput_T,
        	                 &d_WOutput_dZOutput,
        	      	         &d_W2_T,
        	      	         &d_W2_dZ2,
        		         &d_A2_T, &d_A1_T, &d_X_train,
        			 threads_per_block_fp,
        			 number_of_blocks_fp,
        			 sharedMemSize_fp);

        cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

        // Update Parameters
        dim3 threads_per_block5 (16, 16, 1); // A 16 x 16 block threads
        // Remember the testing matrices are 2 x 3
        int N5 = 2;
        dim3 number_of_blocks5 ((N5 / threads_per_block5.x) + 1, (N5 / threads_per_block5.y) + 1, 1);
        update_parameters_GPU <<< threads_per_block5, number_of_blocks5 >>> (d_nn->W1.data, d_nn->b1.data,
        	                                                d_nn->W2.data, d_nn->b2.data,
        							d_nn->WOutput.data, d_nn->bOutput.data,
        							d_dW1.data, d_db1,
        							d_dW2.data, d_db2,
        							d_dWOutput.data, d_dbOutput,
        							d_nn->W1.rows, d_nn->W1.cols,
        							d_nn->b1.rows,
        							d_nn->W2.rows, d_nn->W2.cols,
        							d_nn->b2.rows,
        							d_nn->WOutput.rows, d_nn->WOutput.cols,
        							d_nn->bOutput.rows,
        							d_learning_rate);

        cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

        // Get Predictions
        argmax_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_Y_T.data,
        		                                              d_Y_true.data,
        	                                                      d_Y_T.rows,
        							      d_Y_T.cols);
        cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
        argmax_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_AOutput.data,
        		                                              d_Y_hat.data,
        	                                                      d_AOutput.rows,
        							      d_AOutput.cols);
        cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

        // Initialize float d_accuracy
        float *d_accuracy;
        cudaMalloc((void **)&d_accuracy, sizeof(float));

        // Calculate Accuracy
        calculate_accuracy_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_Y_true.data,
        		                                                          d_Y_hat.data,
        	                                                                  d_Y_true.rows,
        							                  d_accuracy);
        cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

        // Copy float d_accuracy to host
        float h_accuracy;
        cudaMemcpy(&h_accuracy, d_accuracy, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Accuracy: %f\n", h_accuracy);
    }

////////////////////////////////////////////////////////////////////////
// Test CPU Train Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CPU Train Function:\n");

    // Train Neural Network
    train(&nn_value, &X_train, &Y_train, 1, 0.1);

    // Make Predictions
    Matrix Y_pred;
    initialize_matrix(&Y_pred, Y_train.rows, Y_train.cols);
    printf("Making Predictions:\n");
    predict(&nn_value, &X_train, &Y_train, &Y_pred);

    // Compare a few predictions
    denormalize_matrix(&X_train, 0, 255);
    printf("Comparing Predictions:\n");
    preview_predictions(&X_train, &Y_pred, 28, 28, 5);

////////////////////////////////////////////////////////////////////////
// Test GPU Train Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing GPU Train Function:\n");

    // Train Neural Network on device
    dim3 threadsPerBlock (16, 16, 1); // A 16 x 16 block threads
    dim3 numBlocks ((NUM_ROWS_TRAIN / threads_per_block_fp.x) + 1, (NUM_ROWS_TRAIN / threads_per_block_fp.y) + 1, 1);
    int sharedMemSize = sizeof(float) * 32 * 32;
    train_GPU(&d_nn_value, &d_X_train, &d_Y_train, 500, 0.1,
		    threadsPerBlock,
		    numBlocks,
		    sharedMemSize);

////////////////////////////////////////////////////////////////////////
// Free Memory Initiliazed in the Loading Data Section
    // Free X_train
    printf("Freeing X_train on host:\n");
    free_matrix(&X_train);

    // Free d_X_train
    printf("Freeing d_X_train on device:\n");
    free_matrix_on_device(&d_X_train);

    // Free Y_train
    printf("Freeing Y_train on host:\n");
    free_matrix(&Y_train);

    // Free d_Y_train
    printf("Freeing d_Y_train on device:\n");
    free_matrix_on_device(&d_Y_train);

////////////////////////////////////////////////////////////////////////
    // Free Memory Initiliazed in the CPU Forward Propagation Section
    free_neural_network(&nn_value);
    free_matrix(&X_T);
    free_matrix(&Y_T);
    free_matrix(&Z1);
    free_matrix(&A1);
    free_matrix(&Z2);
    free_matrix(&A2);
    free_matrix(&ZOutput);
    free_matrix(&AOutput);
    free_vector(&Y_true);
    free_vector(&Y_hat);
 
////////////////////////////////////////////////////////////////////////
// Free Memory Initiliazed in the CPU Backward Propagation Section
    free_matrix(&dZOutput);
    free_matrix(&dWOutput);
    free_matrix(&A2_T);
    free_matrix(&dZ2);
    free_matrix(&WOutput_T);
    free_matrix(&WOutput_dZOutput);
    free_matrix(&dW2);
    free_matrix(&A1_T);
    free_matrix(&dZ1);
    free_matrix(&W2_T);
    free_matrix(&W2_dZ2);
    free_matrix(&dW1);

////////////////////////////////////////////////////////////////////////
// Free Memory Initiliazed in the GPU Forward Propagation Section
    free_neural_network_on_device(&d_nn_value);
    free_matrix_on_device(&d_X_T);
    free_matrix_on_device(&d_Y_T);
    free_matrix_on_device(&d_Z1);
    free_matrix_on_device(&d_A1);
    free_matrix_on_device(&d_Z2);
    free_matrix_on_device(&d_A2);
    free_matrix_on_device(&d_ZOutput);
    free_matrix_on_device(&d_AOutput);
    free_vector_on_device(&d_Y_true);
    free_vector_on_device(&d_Y_hat);

////////////////////////////////////////////////////////////////////////
// Free Memory Initiliazed in the GPU Backward Propagation Section
    free_matrix_on_device(&d_dZOutput);
    free_matrix_on_device(&d_dWOutput);
    free_matrix_on_device(&d_A2_T);
    free_matrix_on_device(&d_dZ2);
    free_matrix_on_device(&d_WOutput_T);
    free_matrix_on_device(&d_WOutput_dZOutput);
    free_matrix_on_device(&d_dW2);
    free_matrix_on_device(&d_A1_T);
    free_matrix_on_device(&d_dZ1);
    free_matrix_on_device(&d_W2_T);
    free_matrix_on_device(&d_W2_dZ2);
    free_matrix_on_device(&d_dW1);

    return 0;
}
