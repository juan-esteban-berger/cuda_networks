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
// Loading and Preview Functions
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
    // Multiply W1 and X_T
    matrix_multiply(W1, X_T, Z1);
    // printf("Z1:\n");
    // preview_matrix(Z1, 2);

    // Add b1 to Z1
    add_vector_to_matrix(Z1, b1);
    // printf("Z1:\n");
    // preview_matrix(Z1, 2);

    // Apply ReLU to Z1
    ReLU(Z1, A1);
    // printf("A1:\n");
    // preview_matrix(A1, 2);

    // Multiply WOutput and A1
    matrix_multiply(WOutput, A1, ZOutput);
    // printf("ZOutput:\n");
    // preview_matrix(ZOutput, 2);

    // Add bOutput to ZOutput
    add_vector_to_matrix(ZOutput, bOutput);
    // printf("ZOutput:\n");
    // preview_matrix(ZOutput, 2);

    // Apply Softmax to ZOutput
    softmax(ZOutput, AOutput);
    // printf("AOutput:\n");
    // preview_matrix(AOutput, 2);

}

////////////////////////////////////////////////////////////////////////
// Backward Propagation Function
void backward_propagation(Matrix* X_T, Matrix* Y_T,
			  Matrix* W1, Vector* b1,
			  Matrix* WOutput, Vector* bOutput,
			  Matrix* Z1, Matrix* A1,
			  Matrix* ZOutput, Matrix* AOutput,
			  Matrix* dW1, Vector* db1,
			  Matrix* dWOutput, Vector* dbOutput,
			  Matrix* dZ1, Matrix* dZOutput) {
    printf("Backward Propagation:\n");

    // Subtract Y_T from AOutput
    printf("AOutput:\n");
    preview_matrix(AOutput, 2);
    matrix_subtract(AOutput, Y_T, dZOutput);
    preview_matrix(Y_T, 2);
    matrix_subtract(AOutput, Y_T, dZOutput);
    preview_matrix(dZOutput, 2);
}

////////////////////////////////////////////////////////////////////////
// Prediction Function
void predict(Matrix* AOutput, Vector* Y_hat,
	     Matrix* X, int row_index, int image_size_x, int image_size_y) {
    printf("Prediction:\n");

    // Preview the image for the row_index
    printf("Image:\n");
    preview_image(X, row_index, image_size_x, image_size_y);

    // Print the prediction
    argmax(AOutput, Y_hat);
    printf("Prediction: %d\n", (int)Y_hat->data[row_index]);
    printf("\n");
}

////////////////////////////////////////////////////////////////////////
// Training Function
void train(Matrix* X_T, Matrix* Y_T,
	   Matrix* W1, Vector* b1,
	   Matrix* WOutput, Vector* bOutput,
	   Matrix* Z1, Matrix* A1, Matrix* ZOutput, Matrix* AOutput,
	   Matrix* dW1, Vector* db1, Matrix* dWOutput, Vector* dbOutput,
	   Matrix* dZ1, Matrix* dZOutput,
	   int epochs, float learning_rate) {
    printf("Training:\n");

    // Loop over the epochs
    for (int epoch = 0; epoch < epochs; epoch++) {
	printf("Epoch %d:\n", epoch);

	// Forward Propagation
	forward_propagation(X_T,
			W1, b1,
			WOutput, bOutput,
			Z1, A1, ZOutput, AOutput);

	// Backward Propagation
	backward_propagation(X_T, Y_T,
			     W1, b1,
			     WOutput, bOutput,
			     Z1, A1,
			     ZOutput, AOutput,
			     dW1, db1,
			     dWOutput, dbOutput,
			     dZ1, dZOutput);
    }
}


////////////////////////////////////////////////////////////////////////
// Main function
int main() {
    // Seed random number generator
    srand(time(NULL));
    printf("\n");

    // Read in data from X_train.csv
    Matrix X_train;
    initialize_matrix(&X_train, 60000, 784);
    read_csv("X_train.csv", &X_train);
    printf("X_train:\n");
    preview_matrix(&X_train, 2);

    // Read in data from Y_train.csv
    Matrix Y_train;
    initialize_matrix(&Y_train, 60000, 10);
    read_csv("Y_train.csv", &Y_train);
    printf("Y_train:\n");
    preview_matrix(&Y_train, 2);

    // Read in data from X_test.csv
    Matrix X_test;
    initialize_matrix(&X_test, 10000, 784);
    read_csv("X_test.csv", &X_test);
    printf("X_test:\n");
    preview_matrix(&X_test, 2);

    // Read in data from Y_test.csv
    Matrix Y_test;
    initialize_matrix(&Y_test, 10000, 10);
    read_csv("Y_test.csv", &Y_test);
    printf("Y_test:\n");
    preview_matrix(&Y_test, 2);

    // Preview the first image from X_train
    printf("First image from X_train:\n");
    preview_image(&X_train, 0, 28, 28);

    // Transpose and normalize X_train
    Matrix X_train_T;
    initialize_matrix(&X_train_T, X_train.cols, X_train.rows);
    transpose_matrix(&X_train, &X_train_T);
    normalize_matrix(&X_train_T, 0, 255);
    //printf("X_train_T:\n");
    // preview_matrix(&X_train_T, 2);

    // Transpose Y_train
    Matrix Y_train_T;
    initialize_matrix(&Y_train_T, Y_train.cols, Y_train.rows);
    transpose_matrix(&Y_train, &Y_train_T);
    // printf("Y_train_T:\n");
    // preview_matrix(&Y_train_T, 2);

    // Transpose and normalize X_test
    Matrix X_test_T;
    initialize_matrix(&X_test_T, X_test.cols, X_test.rows);
    transpose_matrix(&X_test, &X_test_T);
    normalize_matrix(&X_test_T, 0, 255);
    // printf("X_test_T:\n");
    // preview_matrix(&X_test_T, 2);

    // Transpose Y_test
    Matrix Y_test_T;
    initialize_matrix(&Y_test_T, Y_test.cols, Y_test.rows);
    transpose_matrix(&Y_test, &Y_test_T);
    // printf("Y_test_T:\n");
    // preview_matrix(&Y_test_T, 2);

    // Initialize Layer 1 Weights
    Matrix W1;
    initialize_matrix(&W1, 10, 784);
    random_matrix(&W1);
    // printf("W1:\n");
    // preview_matrix(&W1, 2);

    // Initialize Layer 1 Biases
    Vector b1;
    initialize_vector(&b1, 10);
    random_vector(&b1);
    // printf("b1:\n");
    // preview_vector(&b1, 2);

    // Initialize Output Layer Weights
    Matrix WOutput;
    initialize_matrix(&WOutput, 10, 10);
    random_matrix(&WOutput);
    // printf("WOutput:\n");
    // preview_matrix(&WOutput, 2);

    // Initialize Output Layer Biases
    Vector bOutput;
    initialize_vector(&bOutput, 10);
    random_vector(&bOutput);
    // printf("bOutput:\n");
    // preview_vector(&bOutput, 2);

    // Initialize Z1, A1, ZOutput, AOutput
    Matrix Z1;
    initialize_matrix(&Z1, 10, 60000);
    Matrix A1;
    initialize_matrix(&A1, 10, 60000);
    Matrix ZOutput;
    initialize_matrix(&ZOutput, 10, 60000);
    Matrix AOutput;
    initialize_matrix(&AOutput, 10, 60000);

    // Forward Propagation
    // forward_propagation(&X_train_T,
    // 		    &W1, &b1, &WOutput, &bOutput,
    // 		    &Z1, &A1, &ZOutput, &AOutput);

    // Make Prediction with Untrained Model
    // Vector Y_hat;
    // initialize_vector(&Y_hat, 60000);
    // predict(&AOutput, &Y_hat, &X_train, 0, 28, 28);

    // Initialize dW1, db1,
    // 	          dWOutput, dbOutput,
    //             dZ1, dZOutput
    Matrix dW1;
    initialize_matrix(&dW1, W1.rows, W1.cols);
    Vector db1;
    initialize_vector(&db1, b1.rows);
    Matrix dZ1;
    initialize_matrix(&dZ1, Z1.rows, Z1.cols);
    Matrix dWOutput;
    initialize_matrix(&dWOutput, WOutput.rows, WOutput.cols);
    Vector dbOutput;
    initialize_vector(&dbOutput, bOutput.rows);
    Matrix dZOutput;
    initialize_matrix(&dZOutput, ZOutput.rows, ZOutput.cols);

    // Train Model
    train(&X_train_T, &Y_train_T,
	  &W1, &b1,&WOutput, &bOutput,
	  &Z1, &A1, &ZOutput, &AOutput,
	  &dW1, &db1, &dWOutput, &dbOutput,
          &dZ1, &dZOutput,
	  1, 0.01);


    // Free memory
    free_matrix(&X_train);
    free_matrix(&Y_train);
    free_matrix(&X_test);
    free_matrix(&Y_test);
    free_matrix(&X_train_T);
    free_matrix(&X_test_T);

    return 0;
}
