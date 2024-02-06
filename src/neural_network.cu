#include "neural_network.h"
#include "linear_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////
// Function to Initialize Neural Network
void initialize_neural_network(NeuralNetwork* nn,
			       int input_neurons,
			       int h1_neurons,
			       int h2_neurons,
                               int output_neurons) {
    initialize_matrix(&nn->W1, h1_neurons, input_neurons);
    random_matrix(&nn->W1);

    initialize_vector(&nn->b1, h1_neurons);
    random_vector(&nn->b1);

    initialize_matrix(&nn->W2, h2_neurons, h1_neurons);
    random_matrix(&nn->W2);

    initialize_vector(&nn->b2, h2_neurons);
    random_vector(&nn->b2);

    initialize_matrix(&nn->WOutput, output_neurons, h2_neurons);
    random_matrix(&nn->WOutput);

    initialize_vector(&nn->bOutput, output_neurons);
    random_vector(&nn->bOutput);
}

void initialize_neural_network_on_device(NeuralNetwork_GPU* d_nn,
					 int input_neurons,
					 int h1_neurons,
					 int h2_neurons,
					 int output_neurons) {
    initialize_matrix_on_device(&(d_nn->W1), h1_neurons, input_neurons);

    initialize_vector_on_device(&(d_nn->b1), h1_neurons);

    initialize_matrix_on_device(&(d_nn->W2), h2_neurons, h1_neurons);

    initialize_vector_on_device(&(d_nn->b2), h2_neurons);

    initialize_matrix_on_device(&(d_nn->WOutput), output_neurons, h2_neurons);

    initialize_vector_on_device(&(d_nn->bOutput), output_neurons);
}

////////////////////////////////////////////////////////////////////////
// Function to free memory allocated for Neural Network
void free_neural_network(NeuralNetwork* nn) {
    free_matrix(&nn->W1);
    free_vector(&nn->b1);
    free_matrix(&nn->W2);
    free_vector(&nn->b2);
    free_matrix(&nn->WOutput);
    free_vector(&nn->bOutput);
}

void free_neural_network_on_device(NeuralNetwork_GPU* d_nn) {
    free_matrix_on_device(&(d_nn->W1));

    free_vector_on_device(&(d_nn->b1));

    free_matrix_on_device(&(d_nn->W2));

    free_vector_on_device(&(d_nn->b2));

    free_matrix_on_device(&(d_nn->WOutput));

    free_vector_on_device(&(d_nn->bOutput));
}

////////////////////////////////////////////////////////////////////////
// Function to save model
void save_model(const char* filename, NeuralNetwork* nn) {
    // Open the file
    FILE* file = fopen(filename, "w");

    // Save W1 weights as a flattened row
    for (int i = 0; i < nn->W1.rows; ++i) {
        for (int j = 0; j < nn->W1.cols; ++j) {
            fprintf(file, "%lf", nn->W1.data[i][j]);
            if (j < nn->W1.cols - 1) fprintf(file, ",");
        }
    }
    fprintf(file, "\n");

    // Save b1 biases as a flattened row
    for (int i = 0; i < nn->b1.rows; ++i) {
        fprintf(file, "%lf", nn->b1.data[i]);
        if (i < nn->b1.rows - 1) fprintf(file, ",");
    }
    fprintf(file, "\n");

    // Save W2 weights as a flattened row
    for (int i = 0; i < nn->W2.rows; ++i) {
	for (int j = 0; j < nn->W2.cols; ++j) {
	    fprintf(file, "%lf", nn->W2.data[i][j]);
	    if (j < nn->W2.cols - 1) fprintf(file, ",");
	}
    }
    fprintf(file, "\n");

    // Save b2 biases as a flattened row
    for (int i = 0; i < nn->b2.rows; ++i) {
	fprintf(file, "%lf", nn->b2.data[i]);
	if (i < nn->b2.rows - 1) fprintf(file, ",");
    }
    fprintf(file, "\n");

    // Save WOutput weights as a flattened row
    for (int i = 0; i < nn->WOutput.rows; ++i) {
        for (int j = 0; j < nn->WOutput.cols; ++j) {
            fprintf(file, "%lf", nn->WOutput.data[i][j]);
            if (j < nn->WOutput.cols - 1) fprintf(file, ",");
        }
    }
    fprintf(file, "\n");

    // Save bOutput biases as a flattened row
    for (int i = 0; i < nn->bOutput.rows; ++i) {
        fprintf(file, "%lf", nn->bOutput.data[i]);
        if (i < nn->bOutput.rows - 1) fprintf(file, ",");
    }
    fprintf(file, "\n");

    // Close the file
    fclose(file);
}

////////////////////////////////////////////////////////////////////////
// Function to load model
void load_model(const char* filename, NeuralNetwork* nn) {
    // Open the file
    FILE* file = fopen(filename, "r");

    // Load W1 weights from the first flattened row
    for (int i = 0; i < nn->W1.rows; ++i) {
        for (int j = 0; j < nn->W1.cols; ++j) {
            if (fscanf(file, "%lf,", &nn->W1.data[i][j]) != 1) {
                fprintf(stderr, "Error reading W1 from CSV\n");
                fclose(file);
                return;
            }
        }
    }

    // Load b1 biases from the second flattened row
    for (int i = 0; i < nn->b1.rows; ++i) {
        if (fscanf(file, "%lf,", &nn->b1.data[i]) != 1) {
            fprintf(stderr, "Error reading b1 from CSV\n");
            fclose(file);
            return;
        }
    }

    // Load W2 weights from the third flattened row
    for (int i = 0; i < nn->W2.rows; ++i) {
	for (int j = 0; j < nn->W2.cols; ++j) {
	    if (fscanf(file, "%lf,", &nn->W2.data[i][j]) != 1) {
	 	fprintf(stderr, "Error reading W2 from CSV\n");
		fclose(file);
	 	return;
	    }
	}
    }

    // Load b2 biases from the fourth flattened row
    for (int i = 0; i < nn->b2.rows; ++i) {
	if (fscanf(file, "%lf,", &nn->b2.data[i]) != 1) {
	    fprintf(stderr, "Error reading b2 from CSV\n");
	    fclose(file);
	    return;
	}
    }

    // Load WOutput weights from the fifth flattened row
    for (int i = 0; i < nn->WOutput.rows; ++i) {
        for (int j = 0; j < nn->WOutput.cols; ++j) {
            if (fscanf(file, "%lf,", &nn->WOutput.data[i][j]) != 1) {
                fprintf(stderr, "Error reading WOutput from CSV\n");
                fclose(file);
                return;
            }
        }
    }

    // Load bOutput biases from the sixth flattened row
    for (int i = 0; i < nn->bOutput.rows; ++i) {
        if (fscanf(file, "%lf,", &nn->bOutput.data[i]) != 1) {
            fprintf(stderr, "Error reading bOutput from CSV\n");
            fclose(file);
            return;
        }
    }

    // Close the file
    fclose(file);
}

////////////////////////////////////////////////////////////////////////
// Copy Neural Network from Host to Device
void copy_neural_network_to_device(NeuralNetwork* h_nn,
				   NeuralNetwork_GPU* d_nn) {
    copy_matrix_to_device(&(h_nn->W1), &(d_nn->W1));

    copy_vector_to_device(&(h_nn->b1), &(d_nn->b1));

    copy_matrix_to_device(&(h_nn->W2), &(d_nn->W2));

    copy_vector_to_device(&(h_nn->b2), &(d_nn->b2));

    copy_matrix_to_device(&(h_nn->WOutput), &(d_nn->WOutput));

    copy_vector_to_device(&(h_nn->bOutput), &(d_nn->bOutput));
}

// Copy Neural Network from Device to Host
void copy_neural_network_to_host(NeuralNetwork_GPU* d_nn,
				 NeuralNetwork* h_nn) {
    copy_matrix_to_host(&(h_nn->W1), &(d_nn->W1));

    copy_vector_to_host(&(h_nn->b1), &(d_nn->b1));

    copy_matrix_to_host(&(h_nn->W2), &(d_nn->W2));

    copy_vector_to_host(&(h_nn->b2), &(d_nn->b2));

    copy_matrix_to_host(&(h_nn->WOutput), &(d_nn->WOutput));

    copy_vector_to_host(&(h_nn->bOutput), &(d_nn->bOutput));
}

////////////////////////////////////////////////////////////////////////
// Activation Functions
void ReLU(Matrix* m, Matrix* a) {
    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m->cols; j++) {
	    // Apply ReLU to matrix element
	    a->data[i][j] = fmax(0, m->data[i][j]);
	}
    }
}

// CUDA Kernel for ReLU activation function
__global__ void ReLU_GPU(double *input,
		double *output,
		int rows,
		int cols) {
    // Calculate row and column indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if indices are within matrix bounds
    if (idx < rows && idy < cols) {
	// Calculate index
        int index = idx * cols + idy;
	// Apply ReLU to matrix element
	// fmaxf is a CUDA function for fmax
	// fmax calculates the maximum of two numbers
        output[index] = fmaxf(0, input[index]);
    }
}

void ReLU_derivative(Matrix* m, Matrix* a) {
    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m->cols; j++) {
	    // Apply ReLU derivative to matrix element
	    if (m->data[i][j] > 0) {
		a->data[i][j] = 1;
	    } else {
		a->data[i][j] = 0;
	    }
	}
    }
}

__global__ void ReLU_derivative_GPU(double *input,
				    double *output,
				    int rows,
				    int cols) {
    // Calculate row and column indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if indices are within matrix bounds
    if (idx < rows && idy < cols) {
	// Calculate index
        int index = idx * cols + idy;
	// Apply ReLU derivative to matrix element
	if (input[index] > 0) {
	    output[index] = 1.0f;
	} else {
	    output[index] = 0.0f;
	}
    }
}

void softmax(Matrix* m, Matrix* a) {
    // Loop over the columns
    for (int i = 0; i < m->cols; i++) {
	// Set max_val to negative infinity for numerical stability
        double max_val = -FLT_MAX;
	// Loop through each row
        for (int j = 0; j < m->rows; j++) {
	    // If the current element is greater than max_val
            if (m->data[j][i] > max_val) {
		// Set max_val to the current element
                max_val = m->data[j][i];
            }
        }

        // Calculate the sum of the exponentials
        // for the current column
        double sum = 0;
        for (int j = 0; j < m->rows; j++) {
            sum += exp(m->data[j][i] - max_val);
        }
        // Loop through each row
        for (int j = 0; j < m->rows; j++) {
            // Apply Softmax to matrix element
            a->data[j][i] = exp(m->data[j][i] - max_val) / sum;
        }
    }
}

__global__ void softmax_GPU(double *input, double *output, int rows, int cols) {
    // Calculate column index
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure column index is within matrix bounds
    if (col >= cols) return;

    // Set max_val to negative infinity
    double max_val = -FLT_MAX;
    // Iterate over the rows
    for (int row = 0; row < rows; ++row) {
	// Find the maximum value in the column
        max_val = fmaxf(max_val, input[row * cols + col]);
    }

    // Initialize sum_exp to zero
    double sum_exp = 0.0f;
    // Iterate over the rows
    for (int row = 0; row < rows; ++row) {
	// Calculate the sum of the exponentials in the column
        sum_exp += expf(input[row * cols + col] - max_val);
    }

    // Iterate over the rows
    for (int row = 0; row < rows; ++row) {
	// Apply softmax:
	// Subtract the maximum value from the current element
	// Take the exponential of the result
	// Divide by the sum of the exponentials
        output[row * cols + col] = expf(input[row * cols + col] - max_val) / sum_exp;
    }

}

////////////////////////////////////////////////////////////////////////
// Updated Forward Propagation Function
void forward_propagation(Matrix* X_T,
                         Matrix* W1, Vector* b1,
                         Matrix* W2, Vector* b2,
                         Matrix* WOutput, Vector* bOutput,
                         Matrix* Z1, Matrix* A1,
                         Matrix* Z2, Matrix* A2,
                         Matrix* ZOutput, Matrix* AOutput) {

    // First Layer Dot Products: Z1 = matmul(W1, X_T) + b1
    matrix_multiply(W1, X_T, Z1);
    add_vector_to_matrix(Z1, b1);

    // First Layer Activations: A1 = ReLU(Z1)
    ReLU(Z1, A1);

    // Second Layer Dot Products: Z2 = matmul(W2, A1) + b2
    matrix_multiply(W2, A1, Z2);
    add_vector_to_matrix(Z2, b2);

    // Second Layer Activations: A2 = ReLU(Z2)
    ReLU(Z2, A2);

    // Output Layer Dot Products: ZOutput = matmul(WOutput, A2) + bOutput
    matrix_multiply(WOutput, A2, ZOutput);
    add_vector_to_matrix(ZOutput, bOutput);

    // Output Layer Activations: AOutput = Softmax(ZOutput)
    softmax(ZOutput, AOutput);
}

void forward_propagation_GPU(Matrix_GPU* X_T,
                             Matrix_GPU* W1, Vector* b1,
                             Matrix_GPU* W2, Vector* b2,
                             Matrix_GPU* WOutput, Vector* bOutput,
                             Matrix_GPU* Z1, Matrix_GPU* A1,
                             Matrix_GPU* Z2, Matrix_GPU* A2,
                             Matrix_GPU* ZOutput, Matrix_GPU* AOutput,
			     dim3 threadsPerBlock,
			     dim3 numBlocks,
			     int sharedMemSize) {
////////////////////////////////////////////////////////////////////////
    // First Layer Dot Products: Z1 = matmul(W1, X_T) + b1
    // matrix_multiply_GPU(W1, X_T, Z1);
    matrix_multiply_GPU<<<numBlocks, threadsPerBlock >>>(W1->data,
		    					 X_T->data,
							 Z1->data,
							 W1->rows,
							 W1->cols,
							 X_T->cols);
    cudaDeviceSynchronize();
    // add_vector_to_matrix(Z1, b1);
    add_vector_to_matrix_GPU<<<numBlocks, threadsPerBlock>>>(Z1->data,
							     b1->data,
							     Z1->rows,
							     Z1->cols);

    cudaDeviceSynchronize();
    // First Layer Activations: A1 = ReLU(Z1)
    // ReLU(Z1, A1);
    ReLU_GPU<<<numBlocks, threadsPerBlock>>>(Z1->data,
					     A1->data,
					     Z1->rows,
					     Z1->cols);

    cudaDeviceSynchronize();
////////////////////////////////////////////////////////////////////////
    // Second Layer Dot Products: Z2 = matmul(W2, A1) + b2
    // matrix_multiply(W2, A1, Z2);
    matrix_multiply_GPU<<<numBlocks, threadsPerBlock>>>(W2->data,
		    						       A1->data,
								       Z2->data,
								       W2->rows,
								       W2->cols,
								       A1->cols);
    cudaDeviceSynchronize();
    // add_vector_to_matrix(Z2, b2);
    add_vector_to_matrix_GPU<<<numBlocks, threadsPerBlock>>>(Z2->data,
							     b2->data,
							     Z2->rows,
							     Z2->cols);
    cudaDeviceSynchronize();
    // Second Layer Activations: A2 = ReLU(Z2)
    // ReLU(Z2, A2);
    ReLU_GPU<<<numBlocks, threadsPerBlock>>>(Z2->data,
					     A2->data,
					     Z2->rows,
					     Z2->cols);
    cudaDeviceSynchronize();
////////////////////////////////////////////////////////////////////////
    // Output Layer Dot Products: ZOutput = matmul(WOutput, A2) + bOutput
    // matrix_multiply(WOutput, A2, ZOutput);
    matrix_multiply_GPU<<<numBlocks, threadsPerBlock>>>(WOutput->data,
		    						       A2->data,
								       ZOutput->data,
								       WOutput->rows,
								       WOutput->cols,
								       A2->cols);
    cudaDeviceSynchronize();
    // add_vector_to_matrix(ZOutput, bOutput);
    add_vector_to_matrix_GPU<<<numBlocks, threadsPerBlock>>>(ZOutput->data,
							     bOutput->data,
							     ZOutput->rows,
							     ZOutput->cols);
    cudaDeviceSynchronize();
    // Output Layer Activations: AOutput = Softmax(ZOutput)
    // softmax(ZOutput, AOutput);
    softmax_GPU<<<numBlocks, threadsPerBlock, sharedMemSize>>>(ZOutput->data,
							       AOutput->data,
							       ZOutput->rows,
							       ZOutput->cols);
    cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////
// Backward Propagation Function
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
			  Matrix* A2_T, Matrix* A1_T, Matrix* X) {
////////////////////////////////////////////////////////////////////////
// Output Layer Gradients
    // dZOutput = AOutput - Y_T
    matrix_subtract(AOutput, Y_T, dZOutput);

    // dW2 = 1/m * matmul(dZOutput, A2_T)
    transpose_matrix(A2, A2_T);
    matrix_multiply(dZOutput, A2_T, dWOutput);
    divide_matrix_by_scalar(dWOutput, AOutput->cols);

    // dbOutput = 1/m * sum(dZOutput)
    sum_matrix(dZOutput, dbOutput);
    *dbOutput /= AOutput->cols;

////////////////////////////////////////////////////////////////////////
// Second Layer Gradients
    // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_deriv(Z2)
    transpose_matrix(WOutput, WOutput_T);
    matrix_multiply(WOutput_T, dZOutput, WOutput_dZOutput);
    ReLU_derivative(Z2, dZ2);
    matrix_multiply_elementwise(dZ2, WOutput_dZOutput, dZ2);

    // dW2 = 1 / m * matmul(dZ2, A1_T)
    transpose_matrix(A1, A1_T);
    matrix_multiply(dZ2, A1_T, dW2);
    divide_matrix_by_scalar(dW2, AOutput->cols);

    // db2 = 1/m * sum(dZ2)
    sum_matrix(dZ2, db2);
    *db2 /= AOutput->cols;

////////////////////////////////////////////////////////////////////////
// First Layer Gradients
    // dZ1 = matmul(W2_T, dZ2) * ReLU_deriv(Z1)
    transpose_matrix(W2, W2_T);
    matrix_multiply(W2_T, dZ2, W2_dZ2);
    ReLU_derivative(Z1, dZ1);
    matrix_multiply_elementwise(dZ1, W2_dZ2, dZ1);

    // dW1 = 1 / m * matmul(dZ1, X_T)
    matrix_multiply(dZ1, X, dW1);
    divide_matrix_by_scalar(dW1, AOutput->cols);

    // db1 = 1/m * sum(dZ1)
    sum_matrix(dZ1, db1);
    *db1 /= AOutput->cols;
}

////////////////////////////////////////////////////////////////////////
// GPU Backward Propagation Function
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
			      int sharedMemSize) {
////////////////////////////////////////////////////////////////////////
// Output Layer Gradients
    // dZOutput = AOutput - Y_T
    matrix_subtract_GPU <<< numBlocks, threadsPerBlock >>>(AOutput->data,
							   Y_T->data,
							   dZOutput->data,
							   AOutput->rows,
							   AOutput->cols);
    cudaDeviceSynchronize();

    // dW2 = 1/m * matmul(dZOutput, A2_T)
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>>(A2->data,
							    A2_T->data,
							    A2->rows,
							    A2->cols);
    cudaDeviceSynchronize();
    matrix_multiply_GPU <<< numBlocks, threadsPerBlock >>>(dZOutput->data,
		    					   A2_T->data,
							   dWOutput->data,
							   dZOutput->rows,
							   dZOutput->cols,
							   A2_T->cols);
    cudaDeviceSynchronize();
    divide_matrix_by_scalar_GPU <<< numBlocks, threadsPerBlock >>>(dWOutput->data,
		                                                   AOutput->cols,
								   dWOutput->rows,
								   dWOutput->cols);
    cudaDeviceSynchronize();

    // dbOutput = 1/m * sum(dZOutput)
    sum_matrix_GPU <<< numBlocks, threadsPerBlock, sharedMemSize >>>(dZOutput->data,
        					      *dbOutput, // Pointer to pointer
        					      dZOutput->rows,
        					      dZOutput->cols);
    cudaDeviceSynchronize();
    scalar_division_GPU <<< numBlocks, threadsPerBlock >>>(*dbOutput,
							   AOutput->cols);

////////////////////////////////////////////////////////////////////////
// Second Layer Gradients
    // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_deriv(Z2)
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>>(WOutput->data,
        						    WOutput_T->data,
        						    WOutput->rows,
        						    WOutput->cols);
    cudaDeviceSynchronize();
    matrix_multiply_GPU <<< numBlocks, threadsPerBlock >>>(WOutput_T->data,
        	    					   dZOutput->data,
        						   WOutput_dZOutput->data,
        						   WOutput_T->rows,
        						   WOutput_T->cols,
        						   dZOutput->cols);
    cudaDeviceSynchronize();
    ReLU_derivative_GPU <<< numBlocks, threadsPerBlock >>>(Z2->data,
        						   dZ2->data,
        						   Z2->rows,
        						   Z2->cols);
    cudaDeviceSynchronize();
    matrix_multiply_elementwise_GPU <<< numBlocks, threadsPerBlock >>>(dZ2->data,
        							       WOutput_dZOutput->data,
        							       dZ2->data,
        							       dZ2->rows,
        							       dZ2->cols);
    cudaDeviceSynchronize();

    // dW2 = 1 / m * matmul(dZ2, A1_T)
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>>(A1->data,
        						    A1_T->data,
        						    A1->rows,
        						    A1->cols);
    cudaDeviceSynchronize();
    matrix_multiply_GPU <<< numBlocks, threadsPerBlock >>>(dZ2->data,
        	    					   A1_T->data,
        						   dW2->data,
        						   dZ2->rows,
        						   dZ2->cols,
        						   A1_T->cols);
    cudaDeviceSynchronize();
    divide_matrix_by_scalar_GPU <<< numBlocks, threadsPerBlock >>>(dW2->data,
        	                                                   AOutput->cols,
        							   dW2->rows,
        							   dW2->cols);
    cudaDeviceSynchronize();

    // db2 = 1/m * sum(dZ2)
    sum_matrix_GPU <<< numBlocks, threadsPerBlock, sharedMemSize >>>(dZ2->data,
						      *db2, // Pointer to pointer
						      dZ2->rows,
						      dZ2->cols);
    cudaDeviceSynchronize();
    scalar_division_GPU <<< numBlocks, threadsPerBlock >>>(*db2,
							   AOutput->cols);

////////////////////////////////////////////////////////////////////////
// First Layer Gradients
    // dZ1 = matmul(W2_T, dZ2) * ReLU_deriv(Z1)
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>>(W2->data,
        						    W2_T->data,
        						    W2->rows,
        						    W2->cols);
    cudaDeviceSynchronize();
    matrix_multiply_GPU <<< numBlocks, threadsPerBlock >>>(W2_T->data,
        	    					   dZ2->data,
        						   W2_dZ2->data,
        						   W2_T->rows,
        						   W2_T->cols,
        						   dZ2->cols);
    cudaDeviceSynchronize();
    ReLU_derivative_GPU <<< numBlocks, threadsPerBlock >>>(Z1->data,
        						   dZ1->data,
        						   Z1->rows,
        						   Z1->cols);
    cudaDeviceSynchronize();
    matrix_multiply_elementwise_GPU <<< numBlocks, threadsPerBlock >>>(dZ1->data,
        							       W2_dZ2->data,
        							       dZ1->data,
        							       dZ1->rows,
        							       dZ1->cols);
    cudaDeviceSynchronize();

    // dW1 = 1 / m * matmul(dZ1, X_T)
    matrix_multiply_GPU <<< numBlocks, threadsPerBlock >>>(dZ1->data,
        	    					   X->data,
        						   dW1->data,
        						   dZ1->rows,
        						   dZ1->cols,
        						   X->cols);
    cudaDeviceSynchronize();
    divide_matrix_by_scalar_GPU <<< numBlocks, threadsPerBlock >>>(dW1->data,
        	                                                   AOutput->cols,
        							   dW1->rows,
        							   dW1->cols);
    cudaDeviceSynchronize();

    // db1 = 1/m * sum(dZ1)
    sum_matrix_GPU <<< numBlocks, threadsPerBlock, sharedMemSize >>>(dZ1->data,
						      *db1, // Pointer to pointer
						      dZ1->rows,
						      dZ1->cols);
    cudaDeviceSynchronize();
    scalar_division_GPU <<< numBlocks, threadsPerBlock >>>(*db1,
							   AOutput->cols);
}
////////////////////////////////////////////////////////////////////////
// Update Parameters Function
void update_parameters(Matrix* W1, Vector* b1,
                       Matrix* W2, Vector* b2,
                       Matrix* WOutput, Vector* bOutput,
                       Matrix* dW1, double db1,
                       Matrix* dW2, double db2,
                       Matrix* dWOutput, double dbOutput,
                       double learning_rate) {
    // Update W1
    for (int i = 0; i < W1->rows; ++i) {
        for (int j = 0; j < W1->cols; ++j) {
            W1->data[i][j] -= learning_rate * dW1->data[i][j];
        }
    }

    // Update b1
    for (int i = 0; i < b1->rows; ++i) {
        b1->data[i] -= learning_rate * db1;
    }

    // Update W2
    for (int i = 0; i < W2->rows; ++i) {
        for (int j = 0; j < W2->cols; ++j) {
            W2->data[i][j] -= learning_rate * dW2->data[i][j];
        }
    }

    // Update b2
    for (int i = 0; i < b2->rows; ++i) {
        b2->data[i] -= learning_rate * db2;
    }

    // Update WOutput
    for (int i = 0; i < WOutput->rows; ++i) {
        for (int j = 0; j < WOutput->cols; ++j) {
            WOutput->data[i][j] -= learning_rate * dWOutput->data[i][j];
        }
    }

    // Update bOutput
    for (int i = 0; i < bOutput->rows; ++i) {
        bOutput->data[i] -= learning_rate * dbOutput;
    }
}

__global__ void update_parameters_GPU_old(double *W1, double *b1,
				      double *W2, double *b2,
				      double *WOutput, double *bOutput,
				      double *dW1, double* db1,
				      double *dW2, double* db2,
				      double *dWOutput, double* dbOutput,
				      int W1_rows, int W1_cols,
				      int b1_rows,
				      int W2_rows, int W2_cols,
				      int b2_rows,
				      int WOutput_rows, int WOutput_cols,
				      int bOutput_rows,
				      double learning_rate) {
    // Calculate row and column indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Update W1
    if (idx < W1_rows && idy < W1_cols) {
	W1[idx * W1_cols + idy] -= learning_rate * dW1[idx * W1_cols + idy];
    }

    // Update b1
    if (idx < b1_rows) {
	b1[idx] -= learning_rate * db1[0];
    }

    // Update W2
    if (idx < W2_rows && idy < W2_cols) {
	W2[idx * W2_cols + idy] -= learning_rate * dW2[idx * W2_cols + idy];
    }

    // Update b2
    if (idx < b2_rows) {
	b2[idx] -= learning_rate * db2[0];
    }

    // Update WOutput
    if (idx < WOutput_rows && idy < WOutput_cols) {
	WOutput[idx * WOutput_cols + idy] -= learning_rate * dWOutput[idx * WOutput_cols + idy];
    }

    // Update bOutput
    if (idx < bOutput_rows) {
	bOutput[idx] -= learning_rate * dbOutput[0];
    }
}

#define CLIP_THRESHOLD 1.0f

__global__ void update_parameters_GPU(double *W1, double *b1,
                                               double *W2, double *b2,
                                               double *WOutput, double *bOutput,
                                               double *dW1, double *db1,
                                               double *dW2, double *db2,
                                               double *dWOutput, double *dbOutput,
                                               int W1_rows, int W1_cols,
                                               int b1_rows,
                                               int W2_rows, int W2_cols,
                                               int b2_rows,
                                               int WOutput_rows, int WOutput_cols,
                                               int bOutput_rows,
                                               double learning_rate) {
    // Calculate row and column indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    double gradient;

    // Update W1 with clipping
    if (idx < W1_rows && idy < W1_cols) {
        gradient = dW1[idx * W1_cols + idy];
        gradient = max(-CLIP_THRESHOLD, min(gradient, CLIP_THRESHOLD)); // Clipping
        W1[idx * W1_cols + idy] -= learning_rate * gradient;
    }

    // Update b1 with clipping
    if (idx < b1_rows) {
        gradient = db1[0];
        gradient = max(-CLIP_THRESHOLD, min(gradient, CLIP_THRESHOLD)); // Clipping
        b1[idx] -= learning_rate * gradient;
    }

    // Update W2 with clipping
    if (idx < W2_rows && idy < W2_cols) {
        gradient = dW2[idx * W2_cols + idy];
        gradient = max(-CLIP_THRESHOLD, min(gradient, CLIP_THRESHOLD)); // Clipping
        W2[idx * W2_cols + idy] -= learning_rate * gradient;
    }

    // Update b2 with clipping
    if (idx < b2_rows) {
        gradient = db2[0];
        gradient = max(-CLIP_THRESHOLD, min(gradient, CLIP_THRESHOLD)); // Clipping
        b2[idx] -= learning_rate * gradient;
    }

    // Update WOutput with clipping
    if (idx < WOutput_rows && idy < WOutput_cols) {
        gradient = dWOutput[idx * WOutput_cols + idy];
        gradient = max(-CLIP_THRESHOLD, min(gradient, CLIP_THRESHOLD)); // Clipping
        WOutput[idx * WOutput_cols + idy] -= learning_rate * gradient;
    }

    // Update bOutput with clipping
    if (idx < bOutput_rows) {
        gradient = dbOutput[0];
        gradient = max(-CLIP_THRESHOLD, min(gradient, CLIP_THRESHOLD)); // Clipping
        bOutput[idx] -= learning_rate * gradient;
    }
}


////////////////////////////////////////////////////////////////////////
// Calculate Accuracy Function
void calculate_accuracy(Vector* Y, Vector* Y_hat) {
    // Calculate the number of correct predictions
    int correct_predictions = 0;
    for (int i = 0; i < Y->rows; i++) {
	if (Y->data[i] == Y_hat->data[i]) {
	    correct_predictions++;
	}
    }

    // Calculate the accuracy
    double accuracy = (double)correct_predictions / (double)Y->rows;
    printf("Accuracy: %lf", accuracy);
    printf("\n");
}

__global__ void calculate_accuracy_GPU(double* Y, double* Y_pred, int rows, double* accuracy) {
    // Set correct_count to zero
    int correct_count = 0;
    // Iterate over the rows
    for (int i = 0; i < rows; ++i) {
	// If the difference between the true and predicted values is less than 1e-6
        if (fabsf(Y[i] - Y_pred[i]) < 1e-6) {
	    // 
            ++correct_count;
        }
    }
    // Calculate the accuracy
    *accuracy = ((double)correct_count) / rows;
}

////////////////////////////////////////////////////////////////////////
// Training Function with Struct Input
void train(NeuralNetwork* nn,
	   Matrix* X, Matrix* Y,
	   int epochs, double learning_rate) {

////////////////////////////////////////////////////////////////////////
// Data Preparatation
    // Transpose X to get correct dimensions for matrix multiplication
    Matrix X_T;
    initialize_matrix(&X_T, X->cols, X->rows);
    transpose_matrix(X, &X_T);

    // Transpose Y_T to match AOutput
    Matrix Y_T;
    initialize_matrix(&Y_T, Y->cols, Y->rows);
    transpose_matrix(Y, &Y_T);

////////////////////////////////////////////////////////////////////////
// Initialize Vectors and Matrices needed in Forward Propagation
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

////////////////////////////////////////////////////////////////////////
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
    double dbOutput;

////////////////////////////////////////////////////////////////////////
// Vectors/Matrices Needed for Calculation of Second Layer Gradients
    // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_deriv(Z2)
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
    double db2;

////////////////////////////////////////////////////////////////////////
// Vectors/Matrices Needed for Calculation of First Layer Gradients
    // dZ1 = matmul(WOutput_T, dZ1) * ReLU_deriv(Z1)
    Matrix dZ1;
    initialize_matrix(&dZ1, Z1.rows, Z1.cols);
    Matrix W2_T;
    initialize_matrix(&W2_T, nn->W2.cols, nn->W2.rows);
    Matrix W2_dZ2; // Product of W2_T and dZ2
    initialize_matrix(&W2_dZ2, W2_T.rows, dZ2.cols);

    // dW1 = 1 / m * matmul(dZ1, X_T)
    Matrix dW1;
    initialize_matrix(&dW1, nn->W1.rows, nn->W1.cols);

    // db1 = 1/m * sum(dZ1)
    double db1;

////////////////////////////////////////////////////////////////////////
// Initialize Vectors needed for calculating training accuracy
    Vector Y_true;
    initialize_vector(&Y_true, X_T.cols);
    Vector Y_hat;
    initialize_vector(&Y_hat, X_T.cols);

////////////////////////////////////////////////////////////////////////
// Train Network
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
			     &A2_T, &A1_T, X);

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
// Free Memory
    // Free memory from data preparation section
    free_matrix(&X_T);
    free_matrix(&Y_T);

    // Free memory from forward propagation section
    free_matrix(&Z1);
    free_matrix(&A1);
    free_matrix(&Z2);
    free_matrix(&A2);
    free_matrix(&ZOutput);
    free_matrix(&AOutput);

    // Free memory from output layer gradients calculation
    free_matrix(&dZOutput);
    free_matrix(&dWOutput);
    free_matrix(&A2_T);

    // Free memory from second layer gradients calculation
    free_matrix(&dZ2);
    free_matrix(&WOutput_T);
    free_matrix(&WOutput_dZOutput);
    free_matrix(&dW2);
    free_matrix(&A1_T);

    // Free memory from first layer gradients calculation
    free_matrix(&dZ1);
    free_matrix(&W2_T);
    free_matrix(&W2_dZ2);
    free_matrix(&dW1);

    // Free memory from calculating accuracy section
    free_vector(&Y_true);
    free_vector(&Y_hat);
}

////////////////////////////////////////////////////////////////////////
// GPU Gradient Descent Function
void gradient_descent_GPU(NeuralNetwork_GPU* d_nn,
	       Matrix_GPU* d_X, Matrix_GPU* d_Y,
	       double learning_rate, double* accuracy,
	       dim3 threadsPerBlock,
	       dim3 numBlocks,
	       int sharedMemSize) {

////////////////////////////////////////////////////////////////////////
// Vectors and Matrices Needed for Forward Propagation
    Matrix_GPU d_X_T;
    initialize_matrix_on_device(&d_X_T, d_X->cols, d_X->rows);

    // Transpose Matrix d_X
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>> (d_X->data,
		                                                d_X_T.data,
								d_X->rows,
								d_X->cols);

    cudaDeviceSynchronize();

    // Initialize Matrix d_Y_T on device
    Matrix_GPU d_Y_T;
    initialize_matrix_on_device(&d_Y_T, d_Y->cols, d_Y->rows);

    // Transpose Matrix d_Y
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>> (d_Y->data,
		                                                d_Y_T.data,
								d_Y->rows,
								d_Y->cols);

    cudaDeviceSynchronize();

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
// Vectors/Matrices Needed for Calculation of Output Layer Gradients
    // dZOutput = AOutput - Y_T
    Matrix_GPU d_dZOutput;
    initialize_matrix_on_device(&d_dZOutput, d_ZOutput.rows, d_ZOutput.cols);
    // dWOutput = 1/m * matmul(dZOutput, A2_T)
    Matrix_GPU d_dWOutput;
    initialize_matrix_on_device(&d_dWOutput, d_nn->WOutput.rows, d_nn->WOutput.cols);
    Matrix_GPU d_A2_T;
    initialize_matrix_on_device(&d_A2_T, d_A2.cols, d_A2.rows);
    // dbOutput = 1/m * sum(dZOutput)
    double* d_dbOutput;
    cudaMalloc((void **)&d_dbOutput, sizeof(double));

////////////////////////////////////////////////////////////////////////
    // Vectors/Matrices Needed for Calculation of Second Layer Gradients
    // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_derivative(Z2)
    Matrix_GPU d_dZ2;
    initialize_matrix_on_device(&d_dZ2, d_Z2.rows, d_Z2.cols);
    Matrix_GPU d_WOutput_T;
    initialize_matrix_on_device(&d_WOutput_T, d_nn->WOutput.cols, d_nn->WOutput.rows);
    Matrix_GPU d_WOutput_dZOutput; // Product of WOutput_T and dZOutput
    initialize_matrix_on_device(&d_WOutput_dZOutput, d_WOutput_T.rows, d_dZOutput.cols);
    // dW2 = 1/m * matmul(dZ2, A1_T)
    Matrix_GPU d_dW2;
    initialize_matrix_on_device(&d_dW2, d_nn->W2.rows, d_nn->W2.cols);
    Matrix_GPU d_A1_T;
    initialize_matrix_on_device(&d_A1_T, d_A1.cols, d_A1.rows);
    // db2 = 1/m * sum(dZ2)
    double* d_db2;
    cudaMalloc((void **)&d_db2, sizeof(double));

////////////////////////////////////////////////////////////////////////
// Vectors/Matrices Needed for Calculation of First Layer Gradients
    // dZ1 = matmul(W2_T, dZ2) * ReLU_deriv(Z1)
    Matrix_GPU d_dZ1;
    initialize_matrix_on_device(&d_dZ1, d_Z1.rows, d_Z1.cols);
    Matrix_GPU d_W2_T;
    initialize_matrix_on_device(&d_W2_T, d_nn->W2.cols, d_nn->W2.rows);
    Matrix_GPU d_W2_dZ2; // Product of W2_T and dZ2
    initialize_matrix_on_device(&d_W2_dZ2, d_W2_T.rows, d_dZ2.cols);
    // dW1 = 1/m * matmul(dZ1, X_T)
    Matrix_GPU d_dW1;
    initialize_matrix_on_device(&d_dW1, d_nn->W1.rows, d_nn->W1.cols);
    // db1 = 1/m * sum(dZ1)
    double* d_db1;
    cudaMalloc((void **)&d_db1, sizeof(double));

////////////////////////////////////////////////////////////////////////
// Train Network
    // Forward Propagation
    forward_propagation_GPU(&d_X_T,
    		       &d_nn->W1, &d_nn->b1,
    		       &d_nn->W2, &d_nn->b2,
    		       &d_nn->WOutput, &d_nn->bOutput,
    		       &d_Z1, &d_A1,
    		       &d_Z2, &d_A2,
    		       &d_ZOutput, &d_AOutput,
    		       threadsPerBlock,
    		       numBlocks,
    		       sharedMemSize);
    
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
    			 &d_A2_T, &d_A1_T, d_X,
    			 threadsPerBlock,
    			 numBlocks,
    			 sharedMemSize);
    
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
    
    dim3 numBlocksUp ((2 / threadsPerBlock.x) + 1, (2 / threadsPerBlock.y) + 1, 1);
    update_parameters_GPU <<< threadsPerBlock, numBlocksUp >>> (d_nn->W1.data, d_nn->b1.data,
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
    						learning_rate);
    
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
    
    
    // Get Predictions
    argmax_GPU <<< numBlocks, threadsPerBlock >>> (d_Y_T.data,
    						      d_Y_true.data,
    						      d_Y_T.rows,
    						      d_Y_T.cols);
    
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
    argmax_GPU <<< numBlocks, threadsPerBlock >>> (d_AOutput.data,
    						      d_Y_hat.data,
    						      d_AOutput.rows,
    						      d_AOutput.cols);
    
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
    
    // Initialize double d_accuracy
    double *d_accuracy;
    cudaMalloc((void **)&d_accuracy, sizeof(double));
    
    // Calculate Accuracy
    calculate_accuracy_GPU <<< numBlocks, threadsPerBlock >>> (d_Y_true.data,
    								  d_Y_hat.data,
    								  d_Y_true.rows,
    								  d_accuracy);
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
    
    // Copy double d_accuracy to host
    cudaMemcpy(accuracy, d_accuracy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_accuracy);

    // Free Memory
    free_matrix_on_device(&d_X_T);
    free_matrix_on_device(&d_Y_T);

    free_matrix_on_device(&d_Z1);
    free_matrix_on_device(&d_A1);
    free_matrix_on_device(&d_Z2);
    free_matrix_on_device(&d_A2);
    free_matrix_on_device(&d_ZOutput);
    free_matrix_on_device(&d_AOutput);

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

    free_vector_on_device(&d_Y_true);
    free_vector_on_device(&d_Y_hat);
    cudaFree(d_dbOutput);
    cudaFree(d_db2);
    cudaFree(d_db1);

}

////////////////////////////////////////////////////////////////////////
// Train Network on GPU
void train_GPU(NeuralNetwork_GPU* d_nn,
               Matrix_GPU* d_X, Matrix_GPU* d_Y,
               int epochs, double learning_rate, int batch_size,
               dim3 threadsPerBlock,
               dim3 numBlocks,
               int sharedMemSize) 
{
    // Temporary X Matrix for Mini-Batch Gradient Descent
    Matrix_GPU d_X_subset;
    initialize_matrix_on_device(&d_X_subset, batch_size, d_X->cols);

    // Temporary Y Matrix for Mini-Batch Gradient Descent
    Matrix_GPU d_Y_subset;
    initialize_matrix_on_device(&d_Y_subset, batch_size, d_Y->cols);

    // Initialize training parameters
    double accuracy;
    int epoch = 0;

    // Initialize parameters needed for progress bar
    int total_examples = d_X->rows;
    int total_batches = total_examples / batch_size;
    int batchSize = (int) ceil((double)total_batches / 30.0) * batch_size;

    // Loop over the epochs
    do {
        //clearing accuracy for each new epoch
        accuracy = 0.0;  

        for (int i = 0; i < total_examples; i += batch_size) {
            // Copy a subset of X and Y to d_X_subset and d_Y_subset
            copy_random_matrix_range_to_matrix_GPU(d_X, &d_X_subset,
                                                   d_Y, &d_Y_subset,
                                                   batch_size, total_examples);
            // Perform Gradient Descent on Mini-Batch
            double curr_accuracy;
            gradient_descent_GPU(d_nn, &d_X_subset, &d_Y_subset,
                                   learning_rate, &curr_accuracy,
                                   threadsPerBlock,
                                   numBlocks,
                                   sharedMemSize);
            accuracy += curr_accuracy;

            // Calculate progress
            int progress = (int)(((double)i / total_examples) * 30);

            // Print progress bar
            printf("|");
            for (int p = 0; p < 30; p++) {
                if (p < progress)
                    printf("=");
                else
                    printf(" ");
            }
            printf("| epoch %04d (%06d/%06d) accuracy: %.3lf\r", epoch + 1, i+batch_size, total_examples, accuracy / ((i / batch_size) + 1));
            fflush(stdout);
            }
	    printf("|");
	    for (int p = 0; p < 30; p++) {
		printf("=");
	    }
	    printf("| epoch %04d (%06d/%06d) accuracy: %.3lf\n", epoch + 1, total_examples, total_examples, accuracy / total_batches);
	    fflush(stdout);
            
        epoch++;

    } while (epoch < epochs);

    // Free Memory
    free_matrix_on_device(&d_X_subset);
    free_matrix_on_device(&d_Y_subset);
}

////////////////////////////////////////////////////////////////////////
// Function to make predictions
void predict(NeuralNetwork* nn,
	     Matrix* X, Matrix* Y, Matrix* Y_pred) {
////////////////////////////////////////////////////////////////////////////
// Data Preparation
    // Transpose X to get the correct dimensions for matrix multiplication
    Matrix X_T;
    initialize_matrix(&X_T, X->cols, X->rows);
    transpose_matrix(X, &X_T);

    // Transpose Y_T to match AOutput
    Matrix Y_T;
    initialize_matrix(&Y_T, Y->cols, Y->rows);
    transpose_matrix(Y, &Y_T);

////////////////////////////////////////////////////////////////////////////
// Initialize Vectors and Matrices needed in Forward Propagation
    // Initialize Z1 and A1 used in Forward Propagation
    Matrix Z1;
    initialize_matrix(&Z1, nn->W1.rows, X_T.cols);
    Matrix A1;
    initialize_matrix(&A1, nn->W1.rows, X_T.cols);

    // Initialize Z2 and A2 used in Forward Propagation
    Matrix Z2;
    initialize_matrix(&Z2, nn->W2.rows, X_T.cols);
    Matrix A2;
    initialize_matrix(&A2, nn->W2.rows, X_T.cols);

    // Initialize ZOutput and a temporary AOutput
    Matrix ZOutput;
    initialize_matrix(&ZOutput, nn->WOutput.rows, X_T.cols);
    Matrix AOutput;
    initialize_matrix(&AOutput, nn->WOutput.rows, X_T.cols);

////////////////////////////////////////////////////////////////////////
// Initialize Vectors needed for calculating training accuracy
    // Initialize Vectors for Y and Y_hat
    Vector Y_true;
    initialize_vector(&Y_true, X_T.cols);
    Vector Y_hat;
    initialize_vector(&Y_hat, X_T.cols);

////////////////////////////////////////////////////////////////////////////
// Make Predictions
    // Forward Propagation
    forward_propagation(&X_T,
            &(nn->W1), &(nn->b1),
	    &(nn->W2), &(nn->b2),
            &(nn->WOutput), &(nn->bOutput),
	    &Z1, &A1,
            &Z1, &A1,
	    &ZOutput, &AOutput);

    // Get Predictions
    argmax(&Y_T, &Y_true);
    argmax(&AOutput, &Y_hat);
    
    // Calculate Accuracy
    calculate_accuracy(&Y_true, &Y_hat);

////////////////////////////////////////////////////////////////////////////
// Prepare Predictions
    // Transpose AOutput_tmp into Y_pred
    transpose_matrix(&AOutput, Y_pred);

////////////////////////////////////////////////////////////////////////////
// Free Memory
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
}

////////////////////////////////////////////////////////////////////////
void predict_GPU(NeuralNetwork_GPU* d_nn,
		 Matrix_GPU* d_X, Matrix_GPU* d_Y, Matrix_GPU* d_Y_pred,
		 dim3 threadsPerBlock,
		 dim3 numBlocks,
		 int sharedMemSize) {
////////////////////////////////////////////////////////////////////////////
// Data Preparation
    // Transpose X to get the correct dimensions for matrix multiplication
    Matrix_GPU d_X_T;
    initialize_matrix_on_device(&d_X_T, d_X->cols, d_X->rows);
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>>(d_X->data,
							    d_X_T.data,
							    d_X->rows,
							    d_X->cols);

    cudaDeviceSynchronize();

    // Transpose Y_T to match AOutput
    Matrix_GPU d_Y_T;
    initialize_matrix_on_device(&d_Y_T, d_Y->cols, d_Y->rows);
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>>(d_Y->data,
							    d_Y_T.data,
							    d_Y->rows,
							    d_Y->cols);

    cudaDeviceSynchronize();

////////////////////////////////////////////////////////////////////////////
// Initialize Vectors and Matrices needed in Forward Propagation
    // Initialize Z1 and A1 used in Forward Propagation
    Matrix_GPU d_Z1;
    initialize_matrix_on_device(&d_Z1, d_nn->W1.rows, d_X_T.cols);
    Matrix_GPU d_A1;
    initialize_matrix_on_device(&d_A1, d_nn->W1.rows, d_X_T.cols);

    // Initialize Z2 and A2 used in Forward Propagation
    Matrix_GPU d_Z2;
    initialize_matrix_on_device(&d_Z2, d_nn->W2.rows, d_X_T.cols);
    Matrix_GPU d_A2;
    initialize_matrix_on_device(&d_A2, d_nn->W2.rows, d_X_T.cols);

    // Initialize ZOutput and a temporary AOutput
    Matrix_GPU d_ZOutput;
    initialize_matrix_on_device(&d_ZOutput, d_nn->WOutput.rows, d_X_T.cols);
    Matrix_GPU d_AOutput;
    initialize_matrix_on_device(&d_AOutput, d_nn->WOutput.rows, d_X_T.cols);

////////////////////////////////////////////////////////////////////////
// Initialize Vectors needed for calculating training accuracy
    // Initialize Vectors for Y and Y_hat
    Vector d_Y_true;
    initialize_vector_on_device(&d_Y_true, d_X_T.cols);
    Vector d_Y_hat;
    initialize_vector_on_device(&d_Y_hat, d_X_T.cols);

////////////////////////////////////////////////////////////////////////////
// Make Predictions
    // Forward Propagation
    forward_propagation_GPU(&d_X_T,
	    &(d_nn->W1), &(d_nn->b1),
	    &(d_nn->W2), &(d_nn->b2),
	    &(d_nn->WOutput), &(d_nn->bOutput),
	    &d_Z1, &d_A1,
	    &d_Z1, &d_A1,
	    &d_ZOutput, &d_AOutput,
	    threadsPerBlock,
	    numBlocks,
	    sharedMemSize);

    // Get Predictions
    argmax_GPU <<< numBlocks, threadsPerBlock >>>(d_Y_T.data,
						  d_Y_true.data,
						  d_Y_T.rows,
						  d_Y_T.cols);

    cudaDeviceSynchronize();

    argmax_GPU <<< numBlocks, threadsPerBlock >>>(d_AOutput.data,
		    				  d_Y_hat.data,
						  d_AOutput.rows,
						  d_AOutput.cols);

    cudaDeviceSynchronize();
    
    // Initialize double d_accuracy on device
    double *d_accuracy;
    cudaMalloc((void **)&d_accuracy, sizeof(double));

    // Calculate Accuracy
    calculate_accuracy_GPU <<< numBlocks, threadsPerBlock >>> (d_Y_true.data,
					 d_Y_hat.data,
					 d_Y_true.rows,
					 d_accuracy);
					 

    cudaDeviceSynchronize();

    // Initialize double accuracy on host
    double accuracy;

    // Copy accuracy from device to host
    cudaMemcpy(&accuracy, d_accuracy, sizeof(double), cudaMemcpyDeviceToHost);

    // Print accuracy
    printf("Accuracy: %lf\n", accuracy);

////////////////////////////////////////////////////////////////////////////
// Prepare Predictions
    // Transpose AOutput_tmp into Y_pred
    transpose_matrix_GPU <<< numBlocks, threadsPerBlock >>>(d_AOutput.data,
							    d_Y_pred->data,
							    d_AOutput.rows,
							    d_AOutput.cols);

////////////////////////////////////////////////////////////////////////////
// Free Memory
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

       cudaFree(d_accuracy);
}

////////////////////////////////////////////////////////////////////////
// Functions to compare actual and predicted values
void preview_predictions(Matrix* X, Matrix* Y_pred,
			 int image_size_x, int image_size_y, int n) {
    // Initialize Random Seed
    srand(time(NULL));

    // Repear for the desired number of samples
    for (int i = 0; i < n; i++) {
        // Choose a random row from the dataset
        int random_row = rand() % Y_pred->rows;

        // Display image
        printf("Image at row %d:\n", random_row);
        preview_image(X, random_row, image_size_x, image_size_y);

	// Find the maximum index for each column
        int predicted_class = 0;
        double max_pred_value = Y_pred->data[random_row][0];
        for (int j = 1; j < Y_pred->cols; j++) {
            if (Y_pred->data[random_row][j] > max_pred_value) {
                max_pred_value = Y_pred->data[random_row][j];
                predicted_class = j;
            }
        }

        // Display predicted class
        printf("Predicted digit for image at row %d: %d\n\n",
	       random_row, predicted_class);
    }
}
