#include "linear_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void preview_image(Matrix* m, int row_index,
		   int image_size_x, int image_size_y) {
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
// Normalizing Functions
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

// Function to denormalize a vector
void denormalize_vector(Vector* v, float min, float max) {
    // Iterate over the rows
    for (int i = 0; i < v->rows; i++) {
        // Denormalize vector element
        v->data[i] = v->data[i] * (max - min) + min;
    }
}

// Function to denormalize a matrix
void denormalize_matrix(Matrix* m, float min, float max) {
    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
        // Iterate over the columns
        for (int j = 0; j < m->cols; j++) {
            // Denormalize matrix element
            m->data[i][j] = m->data[i][j] * (max - min) + min;
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Vector and Matrix Operations
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
