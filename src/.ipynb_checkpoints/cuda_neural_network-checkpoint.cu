#include "cuda_neural_network.h"
#include "cuda_linear_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

////////////////////////////////////////////////////////////////////////
// Function to save model
void save_model(const char* filename, NeuralNetwork* nn) {
    // Open the file
    FILE* file = fopen(filename, "w");

    // Save W1 weights as a flattened row
    for (int i = 0; i < nn->W1.rows; ++i) {
        for (int j = 0; j < nn->W1.cols; ++j) {
            fprintf(file, "%f", nn->W1.data[i][j]);
            if (j < nn->W1.cols - 1) fprintf(file, ",");
        }
    }
    fprintf(file, "\n");

    // Save b1 biases as a flattened row
    for (int i = 0; i < nn->b1.rows; ++i) {
        fprintf(file, "%f", nn->b1.data[i]);
        if (i < nn->b1.rows - 1) fprintf(file, ",");
    }
    fprintf(file, "\n");

    // Save W2 weights as a flattened row
    for (int i = 0; i < nn->W2.rows; ++i) {
	for (int j = 0; j < nn->W2.cols; ++j) {
	    fprintf(file, "%f", nn->W2.data[i][j]);
	    if (j < nn->W2.cols - 1) fprintf(file, ",");
	}
    }
    fprintf(file, "\n");

    // Save b2 biases as a flattened row
    for (int i = 0; i < nn->b2.rows; ++i) {
	fprintf(file, "%f", nn->b2.data[i]);
	if (i < nn->b2.rows - 1) fprintf(file, ",");
    }
    fprintf(file, "\n");

    // Save WOutput weights as a flattened row
    for (int i = 0; i < nn->WOutput.rows; ++i) {
        for (int j = 0; j < nn->WOutput.cols; ++j) {
            fprintf(file, "%f", nn->WOutput.data[i][j]);
            if (j < nn->WOutput.cols - 1) fprintf(file, ",");
        }
    }
    fprintf(file, "\n");

    // Save bOutput biases as a flattened row
    for (int i = 0; i < nn->bOutput.rows; ++i) {
        fprintf(file, "%f", nn->bOutput.data[i]);
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
            if (fscanf(file, "%f,", &nn->W1.data[i][j]) != 1) {
                fprintf(stderr, "Error reading W1 from CSV\n");
                fclose(file);
                return;
            }
        }
    }

    // Load b1 biases from the second flattened row
    for (int i = 0; i < nn->b1.rows; ++i) {
        if (fscanf(file, "%f,", &nn->b1.data[i]) != 1) {
            fprintf(stderr, "Error reading b1 from CSV\n");
            fclose(file);
            return;
        }
    }

    // Load W2 weights from the third flattened row
    for (int i = 0; i < nn->W2.rows; ++i) {
	for (int j = 0; j < nn->W2.cols; ++j) {
	    if (fscanf(file, "%f,", &nn->W2.data[i][j]) != 1) {
	 	fprintf(stderr, "Error reading W2 from CSV\n");
		fclose(file);
	 	return;
	    }
	}
    }

    // Load b2 biases from the fourth flattened row
    for (int i = 0; i < nn->b2.rows; ++i) {
	if (fscanf(file, "%f,", &nn->b2.data[i]) != 1) {
	    fprintf(stderr, "Error reading b2 from CSV\n");
	    fclose(file);
	    return;
	}
    }

    // Load WOutput weights from the fifth flattened row
    for (int i = 0; i < nn->WOutput.rows; ++i) {
        for (int j = 0; j < nn->WOutput.cols; ++j) {
            if (fscanf(file, "%f,", &nn->WOutput.data[i][j]) != 1) {
                fprintf(stderr, "Error reading WOutput from CSV\n");
                fclose(file);
                return;
            }
        }
    }

    // Load bOutput biases from the sixth flattened row
    for (int i = 0; i < nn->bOutput.rows; ++i) {
        if (fscanf(file, "%f,", &nn->bOutput.data[i]) != 1) {
            fprintf(stderr, "Error reading bOutput from CSV\n");
            fclose(file);
            return;
        }
    }

    // Close the file
    fclose(file);
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

void softmax(Matrix* m, Matrix* a) {
    // Loop over the columns
    for (int i = 0; i < m->cols; i++) {
	// Calculate the sum of the exponentials
	// for the current column
	float sum = 0;
	for (int j = 0; j < m->rows; j++) {
	    sum += exp(m->data[j][i]);
	}
	// Loop through each row
	for (int j = 0; j < m->rows; j++) {
	    // Apply Softmax to matrix element
	    a->data[j][i] = exp(m->data[j][i]) / sum;
	}
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

////////////////////////////////////////////////////////////////////////
// Backward Propagation Function
void backward_propagation(Matrix* X_T, Matrix* Y_T,
			  Matrix* W1, Vector* b1,
			  Matrix* W2, Vector* b2,
			  Matrix* WOutput, Vector* bOutput,
			  Matrix* Z1, Matrix* A1,
			  Matrix* Z2, Matrix* A2,
			  Matrix* ZOutput, Matrix* AOutput,
			  Matrix* dW1, float* db1,
			  Matrix* dW2, float* db2,
			  Matrix* dWOutput, float* dbOutput,
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
// Update Parameters Function
void update_parameters(Matrix* W1, Vector* b1,
                       Matrix* W2, Vector* b2,
                       Matrix* WOutput, Vector* bOutput,
                       Matrix* dW1, float db1,
                       Matrix* dW2, float db2,
                       Matrix* dWOutput, float dbOutput,
                       float learning_rate) {
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
    float accuracy = (float)correct_predictions / (float)Y->rows;
    printf("Accuracy: %f", accuracy);
    printf("\n");
}

////////////////////////////////////////////////////////////////////////
// Training Function with Struct Input
void train(NeuralNetwork* nn,
	   Matrix* X, Matrix* Y,
	   int epochs, float learning_rate) {

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
    float dbOutput;

////////////////////////////////////////////////////////////////////////
// Vectors/Matrices Needed for Calculation of Second Layer Gradients
    // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_deriv(Z2)
    Matrix dZ2;
    initialize_matrix(&dZ2, Z2.rows, Z2.cols);
    Matrix WOutput_T;
    initialize_matrix(&WOutput_T, nn->WOutput.cols, nn->WOutput.rows);
    Matrix WOutput_dZOutput; // Product of WOutput_T and dZOutput
    initialize_matrix(&WOutput_dZOutput, WOutput_T.rows, dZOutput.cols);

    //dW2 = 1/m * matmul(dZ2, A1_T)
    Matrix dW2;
    initialize_matrix(&dW2, nn->W2.rows, nn->W2.cols);
    Matrix A1_T;
    initialize_matrix(&A1_T, A1.cols, A1.rows);

    // db2 = 1/m * sum(dZ2)
    float db2;

////////////////////////////////////////////////////////////////////////
// Vectors/Matrices Needed for Calculation of First Layer Gradients
    // dZ1 = matmul(WOutput_T, dZOutput) * ReLU_deriv(Z1)
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
    float db1;

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
    free_matrix(&Z1);
    free_matrix(&A1);
    free_matrix(&Z2);
    free_matrix(&A2);
    free_matrix(&ZOutput);
    free_matrix(&AOutput);
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
        float max_pred_value = Y_pred->data[random_row][0];
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
