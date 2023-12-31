#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////
// Structs for vectors and matrices
typedef struct {
    float* data;
    int rows;
} Vector;

typedef struct {
    float** data;
    int rows;
    int cols;
} Matrix;

////////////////////////////////////////////////////////////////////////
// Functions for reading data from csv files
void read_csv(const char* filename, Matrix* matrix) {
    // Open the file with the filename provided.
    // 'r' opens the file for reading.
    FILE* file = fopen(filename, "r");

    // Iterate over the number of rows
    for (int row = 0; row < matrix->rows; ++row) {
        // Iterate over the number of columns
        for (int col = 0; col < matrix->cols; ++col) {
            // Reads a float value from the file and stores
            // it in the data array at the correct position.
	    fscanf(file, "%f,", &matrix->data[row][col]);
        }
    }

    // Close the file
    fclose(file);
}

////////////////////////////////////////////////////////////////////////
// Functions for randomly intializing vectors and matrices
void random_vector(Vector* v) {
    // Iterate over the rows
    for (int i = 0; i < v->rows; i++) {
	// Generate random float between -0.5 and 0.5
	float r = (float)rand() / (float)RAND_MAX - 0.5;

	// Store random float in vector
	v->data[i] = r;
    }
}

void random_matrix(Matrix* m) {
    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m->cols; j++) {
	    // Generate random float between -0.5 and 0.5
	    float r = (float)rand() / (float)RAND_MAX - 0.5;

	    // Store random float in matrix
	    m->data[i][j] = r;
	}
    }
}

////////////////////////////////////////////////////////////////////////
// Functions for previewing data
void preview_vector(Vector* v, int decimals) {
    // Create format string for desired number of decimals
    char format[20];
    sprintf(format, "%%d:\t%%.%df\n", decimals);

    // Print the number of rows
    printf("Vector with %d rows:\n", v->rows);

    // Print single column header
    printf("\t");
    printf("0:\t");
    printf("\n");

    // Iterate over the rows
    for (int i = 0; i < v->rows; i++) {
	// If more than 5 rows, only print first and last 5
	if(i == 5 && v->rows > 5) {
	    printf("...\t...\n");
	    i = v->rows - 5;  // Skip to the last 5 rows
	}
	printf(format, i, v->data[i]);
    }        
    printf("\n");                                   
}

void preview_matrix(Matrix* m, int decimals) {
    // Create format string for desired number of decimals
    char format[20];
    sprintf(format, "%%.%df", decimals);

    // Print the dimensions of the matrix
    printf("Matrix with %d rows and %d columns:\n", m->rows, m->cols);

    // Print column labels
    printf("\t");
    for (int j = 0; j < m->cols; j++) {
        // If more than 8 columns, only print first and last 4
        if(j == 5 && m->cols > 10) {
            printf("...\t");
            j = m->cols - 5;  // Skip to the last 4 columns
        }
        printf("%d: \t", j);
    }
    printf("\n");

    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
        // If more than 10 rows, only print first 5 and last 5
        if(i == 5 && m->rows > 10) {
            printf("...\t");
            for (int k = 0; k < m->cols; k++) {
		// If more than 8 columns, only print first 4 and last 4
		if(k == 5 && m->cols > 10) {
	            printf("...\t");
		    k = m->cols - 5;  // Jump to last 4 columns
		}
		printf("...");
		printf("\t");
            }
            printf("\n");
            i = m->rows - 5;  // Jump to last 5 rows
        }

        // Print row index
        printf("%d:\t", i);

        // Iterate over the columns
        for (int j = 0; j < m->cols; j++) {
            // If more than 8 columns, only print first 4 and last 4
            if(j == 5 && m->cols > 10) {
                printf("...\t");
                j = m->cols - 5;  // Jump to last 4 columns
            }
            printf(format, m->data[i][j]);
            printf("\t");
        }
        printf("\n");
    }
    printf("\n");
}

void preview_image(Matrix* m, int row_index, int image_size_x, int image_size_y) {
    for (int i = 0; i < image_size_x; ++i) {
        for (int j = 0; j < image_size_y; ++j) {
            int value = (int)roundf(m->data[row_index][i*image_size_y + j]);
            if(value == 0) {
                printf("    ");
            } else {
                printf("%03d ", value);
            }
        }
        printf("\n");
    }
    printf("\n");
}

////////////////////////////////////////////////////////////////////////
// Functions for initializing vectors and matrices
void initialize_vector(Vector* v, int rows) {
    // Allocate memory for vector
    v->data = (float*)malloc(rows * sizeof(float));

    // Set number of rows
    v->rows = rows;
}

void initialize_matrix(Matrix* m, int rows, int cols) {
    // Allocate memory for matrix
    m->data = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
	m->data[i] = (float*)malloc(cols * sizeof(float));
    }

    // Set number of rows and columns
    m->rows = rows;
    m->cols = cols;
}

////////////////////////////////////////////////////////////////////////
// Functions to free memory allocated for vectors and matrices
void free_vector(Vector* v) {
    free(v->data);
}

void free_matrix(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
	free(m->data[i]);
    }
    free(m->data);
}

////////////////////////////////////////////////////////////////////////
// Linear Algebra Functions
void normalize_vector(Vector* v, float min, float max) {
    // Iterate over the rows
    for (int i = 0; i < v->rows; i++) {
	// Normalize vector element
	v->data[i] = (v->data[i] - min) / (max - min);
    }
}

void normalize_matrix(Matrix* m, float min, float max) {
    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m->cols; j++) {
	    // Normalize matrix element
	    m->data[i][j] = (m->data[i][j] - min) / (max - min);
	}
    }
}

void transpose_matrix(Matrix* original, Matrix* transpose) {
    // Transpose matrix
    for(int i = 0; i < original->rows; ++i) {
        for(int j = 0; j < original->cols; ++j) {
            transpose->data[j][i] = original->data[i][j];
        }
    }
}

void matrix_multiply(Matrix* m1, Matrix* m2, Matrix* result) {
    // Iterate over the rows
    for(int i = 0; i < m1->rows; ++i) {
	// Iterate over the columns
	for(int j = 0; j < m2->cols; ++j) {
	    // Set result to 0
	    result->data[i][j] = 0;
	    // Calculate dot product for
	    // row i from m1 and column j from m2
	    for(int k = 0; k < m1->cols; ++k) {
		result->data[i][j] += m1->data[i][k] * m2->data[k][j];
	    }
	}
    }
}

void matrix_multiply_elementwise(Matrix* m1, Matrix* m2, Matrix* result) {
    // Iterate over the rows
    for (int i = 0; i < m1->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m1->cols; j++) {
	    // Multiply matrix element by matrix element
	    result->data[i][j] = m1->data[i][j] * m2->data[i][j];
	}
    }
}

void matrix_subtract(Matrix* m1, Matrix* m2, Matrix* result) {
    // Iterate over the rows
    for (int i = 0; i < m1->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m1->cols; j++) {
	    // Subtract matrix element from matrix element
	    result->data[i][j] = m1->data[i][j] - m2->data[i][j];
	}
    }
}

void add_vector_to_matrix(Matrix* m, Vector* v) {
    // Iterate over the columns
    for (int j = 0; j < m->cols; j++) {
	// Iterate over the rows
	for (int i = 0; i < m->rows; i++) {
	    // Add vector element to matrix element
	    m->data[i][j] += v->data[i];
	}
    }
}

void divide_matrix_by_scalar(Matrix* m, float scalar) {
    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m->cols; j++) {
	    // Divide matrix element by scalar
	    m->data[i][j] /= scalar;
	}
    }
}

void sum_matrix(Matrix* m, float* result) {
    // Set result to 0
    *result = 0;
    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m->cols; j++) {
	    // Add matrix element to result
	    *result += m->data[i][j];
	}
    }
}

void argmax(Matrix* m, Vector* v) {
    // Iterate over the columns
    for (int j = 0; j < m->cols; j++) {
	// Set max to first element
	float max = m->data[0][j];
	// Iterate over the rows
	for (int i = 0; i < m->rows; i++) {
	    // If current element is greater than max
	    if (m->data[i][j] > max) {
		// Set max to current element
		max = m->data[i][j];
	    }
	}
	// Iterate over the rows
	for (int i = 0; i < m->rows; i++) {
	    // If current element is equal to max
	    if (m->data[i][j] == max) {
		// Set vector element to row index
		v->data[j] = i;
	    }
	}
    }
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
// Training Function
void train(Matrix* X, Matrix* Y,
	   Matrix* W1, Vector* b1,
	   Matrix* WOutput, Vector* bOutput,
	   int epochs, float learning_rate) {
    printf("Training:\n");

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
    initialize_matrix(&Z1, W1->rows, X_T.cols);
    Matrix A1;
    initialize_matrix(&A1, W1->rows, X_T.cols);

    // Initialize ZOutput and AOutput used in Forward Propagation
    Matrix ZOutput;
    initialize_matrix(&ZOutput, WOutput->rows, X_T.cols);
    Matrix AOutput;
    initialize_matrix(&AOutput, WOutput->rows, X_T.cols);

////////////////////////////////////////////////////////////////////////
// Initialize Vectors and Matrices needed in Backward Propagation
    // Initialize intermediate vars needed for dZOutput calculation
    Matrix dZOutput;
    initialize_matrix(&dZOutput, ZOutput.rows, ZOutput.cols);

    // Initialize intermediate vars needed for dWOutput calculation
    Matrix dWOutput;
    initialize_matrix(&dWOutput, WOutput->rows, WOutput->cols);
    Matrix A1_T;
    initialize_matrix(&A1_T, A1.cols, A1.rows);

    // Initialize intermediate vars needed for dbOutput calculation
    float dbOutput;

    // Initialize intermediate vars needed for dZ1 calculation
    Matrix dZ1;
    initialize_matrix(&dZ1, Z1.rows, Z1.cols);
    Matrix WOutput_T; // Transpose of WOutput
    initialize_matrix(&WOutput_T, WOutput->cols, WOutput->rows);
    Matrix WOutput_dZOutput; // Product of WOutput_T and dZOutput
    initialize_matrix(&WOutput_dZOutput, WOutput_T.rows, dZOutput.cols);
    Matrix Z1_deac; // Z1 with ReLU derivative applied, for backprop
    initialize_matrix(&Z1_deac, Z1.rows, Z1.cols);

    // Initialize intermediate vars needed for dW1 calculation
    Matrix dW1;
    initialize_matrix(&dW1, W1->rows, W1->cols);

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
			W1, b1,
			WOutput, bOutput,
			&Z1, &A1, &ZOutput, &AOutput);

	// Backward Propagation
	backward_propagation(&X_T, &Y_T,
			     W1, b1,
			     WOutput, bOutput,
			     &Z1, &Z1_deac, &A1,
			     &ZOutput, &AOutput,
			     &dW1, &db1,
			     &dWOutput, &dbOutput,
			     &dZ1, &dZOutput,
		      	     &WOutput_T,
		             &WOutput_dZOutput,
			     &A1_T, X);

	// Update Parameters
	update_parameters(W1, b1, WOutput,
		   bOutput, &dW1, db1, &dWOutput,
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

////////////////////////////////////////////////////////////////////////
// Functions to compare actual and predicted values

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
// Load and Preview Data
    // Read in data from X_train.csv
    Matrix X_train;
    initialize_matrix(&X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);
    read_csv("X_train.csv", &X_train);
    printf("X_train:\n");
    preview_matrix(&X_train, 2);

    // Read in data from Y_train.csv
    Matrix Y_train;
    initialize_matrix(&Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);
    read_csv("Y_train.csv", &Y_train);
    printf("Y_train:\n");
    preview_matrix(&Y_train, 2);

    // Read in data from X_test.csv
    Matrix X_test;
    initialize_matrix(&X_test, NUM_ROWS_TEST, NUM_NEURONS_INPUT);
    read_csv("X_test.csv", &X_test);
    printf("X_test:\n");
    preview_matrix(&X_test, 2);

    // Read in data from Y_test.csv
    Matrix Y_test;
    initialize_matrix(&Y_test, NUM_ROWS_TEST, NUM_NEURONS_OUTPUT);
    read_csv("Y_test.csv", &Y_test);
    printf("Y_test:\n");
    preview_matrix(&Y_test, 2);

    // Preview the first image from X_train
    printf("First image from X_train:\n");
    preview_image(&X_train, 0, 28, 28);

    // Normalize X_train and X_test
    normalize_matrix(&X_train, 0, 255);
    normalize_matrix(&X_test, 0, 255);


////////////////////////////////////////////////////////////////////////
// Initialize Weights and Biases (create network struct,
	                       // after organizing code)
    // Initialize Layer 1 Weights
    Matrix W1;
    initialize_matrix(&W1, NUM_NEURONS_HIDDEN_1, NUM_NEURONS_INPUT);
    random_matrix(&W1);

    // Initialize Layer 1 Biases
    Vector b1;
    initialize_vector(&b1, NUM_NEURONS_HIDDEN_1);
    random_vector(&b1);

    // Initialize Output Layer Weights
    Matrix WOutput;
    initialize_matrix(&WOutput, NUM_NEURONS_OUTPUT, NUM_NEURONS_HIDDEN_1);
    random_matrix(&WOutput);

    // Initialize Output Layer Biases
    Vector bOutput;
    initialize_vector(&bOutput, NUM_NEURONS_OUTPUT);
    random_vector(&bOutput);

////////////////////////////////////////////////////////////////////////
// Train Model
    train(&X_train, &Y_train,
	  &W1, &b1, &WOutput, &bOutput,
	  100, 0.1);

////////////////////////////////////////////////////////////////////////
// Test Model (finish organizing code first, then do the testing part...)

////////////////////////////////////////////////////////////////////////
// Free memory
    free_matrix(&X_train);
    free_matrix(&Y_train);
    free_matrix(&X_test);
    free_matrix(&Y_test);

    free_matrix(&W1);
    free_matrix(&WOutput);
    free_vector(&b1);
    free_vector(&bOutput);

    return 0;
}
