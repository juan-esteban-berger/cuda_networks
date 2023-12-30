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
void read_csv(const char* filename, Matrix* matrix, int num_rows, int num_cols) {
    // Open the file with the filename provided.
    // 'r' opens the file for reading.
    FILE* file = fopen(filename, "r");

    // Allocate memory for 2D Array
    float** data = (float**)malloc(num_rows * sizeof(float*));
    for (int row = 0; row < num_rows; ++row) {
        data[row] = (float*)malloc(num_cols * sizeof(float));
    }

    // Iterate over the number of rows
    for (int row = 0; row < num_rows; ++row) {
        // Iterate over the number of columns
        for (int col = 0; col < num_cols; ++col) {
            // Reads a float value from the file and stores
            // it in the data array at the correct position.
            fscanf(file, "%f,", &data[row][col]);
        }
    }

    // Store the data, rows and cols in the Matrix struct
    matrix->data = data;
    matrix->rows = num_rows;
    matrix->cols = num_cols;

    // Close the file
    fclose(file);
}

////////////////////////////////////////////////////////////////////////
// Functions for randomly intializing vectors and matrices
void random_vector(Vector* v) {
    // Allocate memory for vector
    v->data = (float*)malloc(v->rows * sizeof(float));

    // Iterate over the rows
    for (int i = 0; i < v->rows; i++) {
	// Generate random float between -0.5 and 0.5
	float r = (float)rand() / (float)RAND_MAX - 0.5;

	// Store random float in vector
	v->data[i] = r;
    }
}

void random_matrix(Matrix* m) {
    // Allocate memory for matrix
    m->data = (float**)malloc(m->rows * sizeof(float*));
    for (int i = 0; i < m->rows; i++) {
	m->data[i] = (float*)malloc(m->cols * sizeof(float));
    }

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
    // Allocate memory for transposed matrix
    transpose->data = (float**)malloc(original->cols * sizeof(float*));
    for (int i = 0; i < original->cols; i++) {
	transpose->data[i] = (float*)malloc(original->rows * sizeof(float));
    }
    // Set number of rows and columns
    transpose->rows = original->cols;
    transpose->cols = original->rows;

    // Transpose matrix
    for(int i = 0; i < original->rows; ++i) {
        for(int j = 0; j < original->cols; ++j) {
            transpose->data[j][i] = original->data[i][j];
        }
    }
}

void matrix_multiply(Matrix* m1, Matrix* m2, Matrix* result) {
    // Allocate memory for result matrix
    result->data = (float**)malloc(m1->rows * sizeof(float*));
    for (int i = 0; i < m1->rows; i++) {
	result->data[i] = (float*)malloc(m2->cols * sizeof(float));
    }

    // Set number of rows and columns
    result->rows = m1->rows;
    result->cols = m2->cols;

    // Multiply matrices
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

////////////////////////////////////////////////////////////////////////
// Activation Functions
void ReLU(Matrix* m, Matrix* a) {
    // Allocate memory for result matrix
    a->data = (float**)malloc(m->rows * sizeof(float*));
    for (int i = 0; i < m->rows; i++) {
	a->data[i] = (float*)malloc(m->cols * sizeof(float));
    }

    // Set number of rows and columns
    a->rows = m->rows;
    a->cols = m->cols;

    // Apply ReLU to matrix
    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
	// Iterate over the columns
	for (int j = 0; j < m->cols; j++) {
	    // Apply ReLU to matrix element
	    a->data[i][j] = fmax(0, m->data[i][j]);
	}
    }
}

////////////////////////////////////////////////////////////////////////
// Forward Propagation Function
void forward_propagation(Matrix* X_T,
		Matrix* W1, Vector* b1,
		Matrix* WOutput, Vector* bOutput,
		Matrix* Z1, Matrix* A1, Matrix* ZOutput, Matrix* AOutput) {
    printf("Forward Propagation:\n");

    // Multiply W1 and X_T
    matrix_multiply(W1, X_T, Z1);
    printf("Z1:\n");
    preview_matrix(Z1, 2);

    // Add b1 to Z1
    add_vector_to_matrix(Z1, b1);
    printf("Z1:\n");
    preview_matrix(Z1, 2);

    // Apply ReLU to Z1
    ReLU(Z1, A1);
    printf("A1:\n");
    preview_matrix(A1, 2);

    // Multiply WOutput and A1
    matrix_multiply(WOutput, A1, ZOutput);
    printf("ZOutput:\n");
    preview_matrix(ZOutput, 2);

    // Add bOutput to ZOutput
    add_vector_to_matrix(ZOutput, bOutput);
    printf("ZOutput:\n");
    preview_matrix(ZOutput, 2);

    // Apply Softmax to ZOutput

}

////////////////////////////////////////////////////////////////////////
// Main function
int main() {
    // Seed random number generator
    srand(time(NULL));
    printf("\n");

    // Read in data from X_train.csv
    Matrix X_train;
    read_csv("X_train.csv", &X_train, 60000, 784);
    printf("X_train:\n");
    preview_matrix(&X_train, 2);

    // Read in data from Y_train.csv
    Matrix Y_train;
    read_csv("Y_train.csv", &Y_train, 60000, 10);
    printf("Y_train:\n");
    preview_matrix(&Y_train, 2);

    // Read in data from X_test.csv
    Matrix X_test;
    read_csv("X_test.csv", &X_test, 10000, 784);
    printf("X_test:\n");
    preview_matrix(&X_test, 2);

    // Read in data from Y_test.csv
    Matrix Y_test;
    read_csv("Y_test.csv", &Y_test, 10000, 10);
    printf("Y_test:\n");
    preview_matrix(&Y_test, 2);

    // Preview the first image from X_train
    printf("First image from X_train:\n");
    preview_image(&X_train, 0, 28, 28);

    // Transpose and normalize X_train
    Matrix X_train_T;
    transpose_matrix(&X_train, &X_train_T);
    normalize_matrix(&X_train_T, 0, 255);
    printf("X_train_T:\n");
    preview_matrix(&X_train_T, 2);

    // Transpose and normalize X_test
    Matrix X_test_T;
    transpose_matrix(&X_test, &X_test_T);
    normalize_matrix(&X_test_T, 0, 255);
    printf("X_test_T:\n");
    preview_matrix(&X_test_T, 2);

    // Initialize Layer 1 Weights
    Matrix W1;
    W1.rows = 10;
    W1.cols = 784;
    random_matrix(&W1);
    printf("W1:\n");
    preview_matrix(&W1, 2);

    // Initialize Layer 1 Biases
    Vector b1;
    b1.rows = 10;
    random_vector(&b1);
    printf("b1:\n");
    preview_vector(&b1, 2);

    // Initialize Output Layer Weights
    Matrix WOutput;
    WOutput.rows = 10;
    WOutput.cols = 10;
    random_matrix(&WOutput);
    printf("WOutput:\n");
    preview_matrix(&WOutput, 2);

    // Initialize Output Layer Biases
    Vector bOutput;
    bOutput.rows = 10;
    random_vector(&bOutput);
    printf("bOutput:\n");
    preview_vector(&bOutput, 2);

    // Initialize Z1, A1, ZOutput, AOutput
    Matrix Z1;
    Matrix A1;
    Matrix ZOutput;
    Matrix AOutput;

    // Forward Propagation
    forward_propagation(&X_train_T, &W1, &b1, &WOutput, &bOutput, &Z1, &A1, &ZOutput, &AOutput);


    // Free memory
    free_matrix(&X_train);
    free_matrix(&Y_train);
    free_matrix(&X_test);
    free_matrix(&Y_test);
    free_matrix(&X_train_T);
    free_matrix(&X_test_T);

    return 0;
}
