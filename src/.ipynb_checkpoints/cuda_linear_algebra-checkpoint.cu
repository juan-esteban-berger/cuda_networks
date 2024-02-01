#include "cuda_linear_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

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

void initialize_vector_on_device(Vector* v, int rows) {
    // Allocate memory for vector
    cudaMalloc((void**) &v->data, rows * sizeof(float));
    // Set number of rows (stored on host)
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

void initialize_matrix_on_device(Matrix_GPU* m, int rows, int cols) {
    // Calculate the total number of elements in the matrix
    int total_elements = rows * cols;

    // Allocate memory for the matrix elements on the device
    cudaMalloc((void**) &m->data, total_elements * sizeof(float));

    // Set the number of rows and columns in the host memory
    m->rows = rows;
    m->cols = cols;
}

////////////////////////////////////////////////////////////////////////
// Functions to copy vectors and matrices from host to device
void copy_vector_to_device(Vector* h_v, Vector* d_v) {
   cudaMemcpy(d_v->data,
              h_v->data,
              h_v->rows * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_matrix_to_device(Matrix* h_m, Matrix_GPU* d_m) {
    // Calculate the size of a single row in bytes
    size_t row_size = h_m->cols * sizeof(float);

    // Pointer to keep track of where to copy next in the device memory
    float* d_ptr = d_m->data;

    // Copy each row from the host matrix to the device matrix
    for (int i = 0; i < h_m->rows; i++) {
        cudaMemcpy(d_ptr, h_m->data[i], row_size, cudaMemcpyHostToDevice);
        d_ptr += h_m->cols; // Move to the next row in the device memory
    }
}

////////////////////////////////////////////////////////////////////////
// Functions to copy vectors and matrices from device to host
void copy_vector_to_host(Vector* h_v, Vector* d_v) {
   cudaMemcpy(h_v->data,
              d_v->data,
              d_v->rows * sizeof(float), cudaMemcpyDeviceToHost);
}

void copy_matrix_to_host(Matrix* h_m, Matrix_GPU* d_m) {
    // Calculate the size of a single row in bytes
    size_t row_size = h_m->cols * sizeof(float);

    // Pointer to keep track of where to copy from in the device memory
    float* d_ptr = d_m->data;

    // Copy each row from the device matrix to the host matrix
    for (int i = 0; i < h_m->rows; i++) {
        cudaMemcpy(h_m->data[i], d_ptr, row_size, cudaMemcpyDeviceToHost);
        d_ptr += h_m->cols; // Move to the next row in the device memory
    }
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
    v->data = NULL;
}

void free_vector_on_device(Vector* v) {
    cudaFree(v->data);
    v->data = NULL;
}

void free_matrix(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
	free(m->data[i]);
	m->data[i] = NULL;
    }
    free(m->data);
    m->data = NULL;
}

void free_matrix_on_device(Matrix_GPU* m) {
    cudaFree(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
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

__global__ void transpose_matrix_GPU(float *input,
				     float *output,
				     int rows,
				     int cols) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the bounds of the matrix
    if (row < rows && col < cols) {
        // Perform the transpose operation
        output[col * rows + row] = input[row * cols + col];
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

__global__ void matrix_multiply_GPU(float *a,
				    float *b,
				    float *c,
				    int aRows, 
				    int aCols, 
				    int bCols) {

    // Initialize float to store dot product intermediate result
    float val = 0;

    // Calculate row and column indices
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if row and column are within matrix bounds
    if (row < aRows && col < bCols) {
	// Calculate dot product for row of a and column of b
        for (int k = 0; k < aCols; ++k)
            val += a[row * aCols + k] * b[k * bCols + col];
        c[row * bCols + col] = val;
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

__global__ void matrix_multiply_elementwise_GPU(float *a,
						float *b,
						float *c,
						int rows,
						int cols) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate index of current element
    int index = row * cols + col;

    // Check if the thread is within the bounds of the matrix
    if (row < rows && col < cols) {
        // Perform element-wise multiplication
        c[index] = a[index] * b[index];
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

__global__ void matrix_subtract_GPU(float *a,
				    float *b,
				    float *c,
				    int rows,
				    int cols) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate index of current element
    int index = row * cols + col;

    // Check if the thread is within the bounds of the matrix
    if (row < rows && col < cols) {
        // Perform element-wise subtraction
        c[index] = a[index] - b[index];
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

__global__ void divide_matrix_by_scalar_GPU(float *m,
					    float scalar,
					    int rows,
					    int cols) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate index of current element
    int index = row * cols + col;

    // Check if the thread is within the bounds of the matrix
    if (row < rows && col < cols) {
        // Perform division of matrix element by scalar
        m[index] /= scalar;
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

__global__ void sum_matrix_GPU(float *matrix, float *result, int rows, int cols) {
    // Calculate the thread's unique index within the grid.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Declare shared memory dynamically. Shared memory is used to store intermediate
    // sums and is accessible by all threads within the same block.
    extern __shared__ float sdata[];

    // Check if the thread is less than the number of elements in the matrix
    if (index < rows * cols) {
	// Load the element from global memory into shared memory
        sdata[threadIdx.x] = matrix[index];
    } else {
	// If the thread is outside the bounds of the matrix, set the element
	// in shared memory to 0.
        sdata[threadIdx.x] = 0.0f;
    }
    // Ensure that all threads in the block have loaded their elements into shared
    // memory before continuing
    __syncthreads();

    // Perform block-level reduction in shared memory.
    // Iterate over the elements in shared memory, halving the stride at each step.
    // sdata[0] will end up containing the sum of all elements in the block.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
	// Check if the thread is within the stride
        if (threadIdx.x < s) {
	    // Add the element at the current index to the element at the current
	    // index plus the stride
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
	// Ensure that all threads in the block have updated their elements in shared
	// memory before continuing
        __syncthreads();
    }

    // Check if the thread is the first thread in the block
    if (threadIdx.x == 0) {
	// Add the block's sum to the result in global memory
        atomicAdd(result, sdata[0]);
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

__global__ void add_vector_to_matrix_GPU(float *m,
					 float *vector,
					 int rows,
					 int cols) {
    // Calculate row and column indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the matrix
    if (row < rows && col < cols) {
	// Add vector element to matrix element
        m[row * cols + col] += vector[row];
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

__global__ void argmax_GPU(float *m, float *result, int rows, int cols) {
    // Calculate the column index that this thread will operate on
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the column index is within bounds
    if (col < cols) {
        float max_val = m[col];
	// Initialize max index to 0
        int max_idx = 0;

        // Loop over all elements in the column to find the max value
        for (int row = 1; row < rows; row++) {
	    // Get the value at the current index
            float val = m[row * cols + col];
	    // If the value is greater than the current max value
            if (val > max_val) {
		// Set the max value to the current value
                max_val = val;
		// Set the max index to the current row
                max_idx = row;
            }
        }

        // Write the index of the max value to the result vector
        result[col] = max_idx;
    }
}
