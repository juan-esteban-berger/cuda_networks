#include "neural_network.h"
#include "linear_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////
// Function to Initialize Neural Network
void initialize_neural_network(NeuralNetwork* nn,
			       int input_neurons,
			       int hidden_neurons,
                               int output_neurons) {
    initialize_matrix(&nn->W1, hidden_neurons, input_neurons);
    random_matrix(&nn->W1);

    initialize_vector(&nn->b1, hidden_neurons);
    random_vector(&nn->b1);

    initialize_matrix(&nn->WOutput, output_neurons, hidden_neurons);
    random_matrix(&nn->WOutput);

    initialize_vector(&nn->bOutput, output_neurons);
    random_vector(&nn->bOutput);
}

////////////////////////////////////////////////////////////////////////
// Function to free memory allocated for Neural Network
void free_neural_network(NeuralNetwork* nn) {
    free_matrix(&nn->W1);
    free_vector(&nn->b1);
    free_matrix(&nn->WOutput);
    free_vector(&nn->bOutput);
}

////////////////////////////////////////////////////////////////////////
// Function to save model
void save_model(const char* filename, NeuralNetwork* nn) {
    // Open the file
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file to save neural network");
        return;
    }

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
    if (!file) {
        perror("Error opening file to load neural network");
        return;
    }

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

    // Load WOutput weights from the third flattened row
    for (int i = 0; i < nn->WOutput.rows; ++i) {
        for (int j = 0; j < nn->WOutput.cols; ++j) {
            if (fscanf(file, "%f,", &nn->WOutput.data[i][j]) != 1) {
                fprintf(stderr, "Error reading WOutput from CSV\n");
                fclose(file);
                return;
            }
        }
    }

    // Load bOutput biases from the fourth flattened row
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
// Forward Propagation Function
void forward_propagation(Matrix* X_T,
		Matrix* W1, Vector* b1,
		Matrix* WOutput, Vector* bOutput,
		Matrix* Z1, Matrix* A1, Matrix* ZOutput, Matrix* AOutput) {

    // First Layer:
    // Z1 = matmul(W1, X_T) + b
    // A1 = ReLU(Z1)
    matrix_multiply(W1, X_T, Z1);
    add_vector_to_matrix(Z1, b1);
    ReLU(Z1, A1);

    // Output Layer:
    // ZOutput = matmul(WOutput, A1) + bOutput
    // AOutput = Softmax(ZOutput)
    matrix_multiply(WOutput, A1, ZOutput);
    add_vector_to_matrix(ZOutput, bOutput);
    softmax(ZOutput, AOutput);
}

////////////////////////////////////////////////////////////////////////
// Backward Propagation Function
void backward_propagation(Matrix* X_T, Matrix* Y_T,
			  Matrix* W1, Vector* b1,
			  Matrix* WOutput, Vector* bOutput,
			  Matrix* Z1, Matrix* Z1_deac, Matrix* A1,
			  Matrix* ZOutput, Matrix* AOutput,
			  Matrix* dW1, float* db1,
			  Matrix* dWOutput, float* dbOutput,
			  Matrix* dZ1, Matrix* dZOutput,
			  Matrix* WOutput_T,
			  Matrix* WOutput_dZOutput,
			  Matrix* A1_T, Matrix* X) {

    // Derivative of loss with respect to ZOutput
    // Loss: Categorical Cross-Entropy
    // Last Layer Activation: Softmax
    // dZOutput = AOutput - Y_T
    matrix_subtract(AOutput, Y_T, dZOutput);

    // Derivative of loss with respect to WOutput
    // dW2 = 1/m * matmul(dZOutput, A1_T)
    transpose_matrix(A1, A1_T);
    matrix_multiply(dZOutput, A1_T, dWOutput);
    divide_matrix_by_scalar(dWOutput, AOutput->cols);

    // Derivative of loss with respect to bOutput
    // dbOutput = 1/m * sum(dZ2)
    sum_matrix(dZOutput, dbOutput);
    *dbOutput /= AOutput->cols;

    // Derivative of loss with respect to Z1 
    // dZ1 = matmul(WOutput_T, dZOutput) * ReLU_deriv(Z1)
    transpose_matrix(WOutput, WOutput_T);
    matrix_multiply(WOutput_T, dZOutput, WOutput_dZOutput);
    ReLU_derivative(Z1, Z1_deac);
    matrix_multiply_elementwise(Z1_deac, WOutput_dZOutput, dZ1);

    // Derivative of loss with respect to W1
    // dW1 = 1 / m * matmul(dZ1, X_T)
    matrix_multiply(dZ1, X, dW1);
    divide_matrix_by_scalar(dW1, AOutput->cols);

    // Derivative of loss with respect to b1
    // db1 = 1/m * sum(dZ1)
    sum_matrix(dZ1, db1);
    *db1 /= AOutput->cols;
}

////////////////////////////////////////////////////////////////////////
// Update Parameters Function
void update_parameters(Matrix* W1, Vector* b1,
		       Matrix* W2, Vector* b2,
		       Matrix* dW1, float db1,
		       Matrix* dW2, float db2,
                       float learning_rate) {
    // Update W1
    for (int i = 0; i < W1->rows; ++i) {
        for (int j = 0; j < W1->cols; ++j) {
            // W1[i][j] = W1[i][j] - learning_rate * dW1[i][j]
            W1->data[i][j] = W1->data[i][j] - learning_rate * dW1->data[i][j];
        }
    }

    // Update b1
    for (int i = 0; i < b1->rows; ++i) {
        // b1[i] = b1[i] - learning_rate * db1
        b1->data[i] = b1->data[i] - learning_rate * db1;
    }

    // Update W2
    for (int i = 0; i < W2->rows; ++i) {
        for (int j = 0; j < W2->cols; ++j) {
            // W2[i][j] = W2[i][j] - learning_rate * dW2[i][j]
            W2->data[i][j] = W2->data[i][j] - learning_rate * dW2->data[i][j];
        }
    }

    // Update b2
    for (int i = 0; i < b2->rows; ++i) {
        // b2[i] = b2[i] - learning_rate * db2
        b2->data[i] = b2->data[i] - learning_rate * db2;
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
    // Initialize Z1 and A1 used in Forward Propagation
    Matrix Z1;
    initialize_matrix(&Z1, nn->W1.rows, X_T.cols);
    Matrix A1;
    initialize_matrix(&A1, nn->W1.rows, X_T.cols);

    // Initialize ZOutput and AOutput used in Forward Propagation
    Matrix ZOutput;
    initialize_matrix(&ZOutput, nn->WOutput.rows, X_T.cols);
    Matrix AOutput;
    initialize_matrix(&AOutput, nn->WOutput.rows, X_T.cols);

////////////////////////////////////////////////////////////////////////
// Initialize Vectors and Matrices needed in Backward Propagation
    // Initialize intermediate vars needed for dZOutput calculation
    Matrix dZOutput;
    initialize_matrix(&dZOutput, ZOutput.rows, ZOutput.cols);

    // Initialize intermediate vars needed for dWOutput calculation
    Matrix dWOutput;
    initialize_matrix(&dWOutput, nn->WOutput.rows, nn->WOutput.cols);
    Matrix A1_T;
    initialize_matrix(&A1_T, A1.cols, A1.rows);

    // Initialize intermediate vars needed for dbOutput calculation
    float dbOutput;

    // Initialize intermediate vars needed for dZ1 calculation
    Matrix dZ1;
    initialize_matrix(&dZ1, Z1.rows, Z1.cols);
    Matrix WOutput_T; // Transpose of WOutput
    initialize_matrix(&WOutput_T, nn->WOutput.cols, nn->WOutput.rows);
    Matrix WOutput_dZOutput; // Product of WOutput_T and dZOutput
    initialize_matrix(&WOutput_dZOutput, WOutput_T.rows, dZOutput.cols);
    Matrix Z1_deac; // Z1 with ReLU derivative applied, for backprop
    initialize_matrix(&Z1_deac, Z1.rows, Z1.cols);

    // Initialize intermediate vars needed for dW1 calculation
    Matrix dW1;
    initialize_matrix(&dW1, nn->W1.rows, nn->W1.cols);

    // Initialize intermediate vars needed for db1 calculation
    float db1;

////////////////////////////////////////////////////////////////////////
// Initialize Vectors needed for calculating training accuracy
    // Initialize Vectors for Y and Y_hat
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
			&(nn->WOutput), &(nn->bOutput),
			&Z1, &A1, &ZOutput, &AOutput);

	// Backward Propagation
	backward_propagation(&X_T, &Y_T,
			     &(nn->W1), &(nn->b1),
			     &(nn->WOutput), &(nn->bOutput),
			     &Z1, &Z1_deac, &A1,
			     &ZOutput, &AOutput,
			     &dW1, &db1,
			     &dWOutput, &dbOutput,
			     &dZ1, &dZOutput,
		      	     &WOutput_T,
		             &WOutput_dZOutput,
			     &A1_T, X);

	// Update Parameters
	update_parameters(&(nn->W1), &(nn->b1), &(nn->WOutput),
		   &(nn->bOutput), &dW1, db1, &dWOutput,
		   dbOutput, learning_rate);

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
    free_matrix(&ZOutput);
    free_matrix(&AOutput);

    // Free memory from backward propagation section
    free_matrix(&dZOutput);
    free_matrix(&dWOutput);
    free_matrix(&A1_T);
    free_matrix(&dZ1);
    free_matrix(&WOutput_T);
    free_matrix(&WOutput_dZOutput);
    free_matrix(&Z1_deac);
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
            &(nn->WOutput), &(nn->bOutput),
            &Z1, &A1, &ZOutput, &AOutput);

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
