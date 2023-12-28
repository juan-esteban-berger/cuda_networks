#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////////////////
// Loading and Preview Functions
float** read_csv(const char* filename, int num_rows, int num_cols) {
    // Open the file with the filename provided. 'r' opens the file for reading.
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

    // Close the file
    fclose(file);

    // Return the pointer to the array
    return data;
}

void preview_data_1D(float* data, int num_rows, int decimals) {
    float value;
    float round_precision = powf(10, decimals);
    for (int i = 0; i < num_rows; ++i) {
        value = roundf(data[i] * round_precision) / round_precision;
        printf("%.*f ", decimals, value);
    }
    printf("\n");
    printf("\n");
}

void preview_data_2D(float** data, int num_rows, int num_cols, int decimals) {
    float value;
    float round_precision = powf(10, decimals);
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            value = roundf(data[i][j] * round_precision) / round_precision;
            printf("%.*f ", decimals, value);
        }
        printf("\n");
    }
    printf("\n");
}

void preview_image(float** X_train, int index, int image_size_x, int image_size_y) {
    for (int i = 0; i < image_size_x; ++i) {
        for (int j = 0; j < image_size_y; ++j) {
            int value = (int)roundf(X_train[index][i*image_size_y + j]);
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

////////////////////////////////////////////////////////////////////////////////////
// Linear Algebra Functions
void matrix_multiply(float** A, float** B, float** C,
		int A_rows, int A_cols, int B_cols) {
    // Iterate over the number of rows of matrix A
    for (int i = 0; i < A_rows; ++i) {
	// Iterate over the number of columns of matrix B
        for (int j = 0; j < B_cols; ++j) {
	    // Initialize the current element of C to zero
            C[i][j] = 0;
        }
    }

    // Iterate over the number of rows of matrix A
    for (int i = 0; i < A_rows; ++i) {
	// Iterate over the number of columns of matrix B
        for (int j = 0; j < B_cols; ++j) {
	    // Iterate over the number of columns of matrix A
            for (int k = 0; k < A_cols; ++k) {
		// Take the dot product of the current row of A and the current column of B
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void add_vector_to_matrix(float** matrix, float* vec, int rows, int cols) {   
    // Iterate over columns
    for (int j = 0; j < cols; ++j) {
	// Iterate over rows
        for (int i = 0; i < rows; ++i) {
	    // Add corresponding element
            matrix[i][j] += vec[j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// Initialize Parameters
void init_params(int num_rows_w1, int num_cols_w1,
		int num_rows_w2, int num_cols_w2,
		float** W1, float* b1,
		float** W2, float* b2) {
    // Seed Random Number Generator with Current Time
    srand(time(0));

    // Initialize Layer 1 Weights
    for (int i = 0; i < num_rows_w1; ++i) {
        for (int j = 0; j < num_cols_w1; ++j) {
            W1[i][j] = (rand() / (float)RAND_MAX) - 0.5;
        }
    }

    // Initialize Layer 1 Biases
    for (int i = 0; i < num_rows_w1; ++i) {
        b1[i] = (rand() / (float)RAND_MAX) - 0.5;
    }

    // Initialize Layer 2 Weights
    for (int i = 0; i < num_rows_w2; ++i) {
        for (int j = 0; j < num_cols_w2; ++j) {
            W2[i][j] = (rand() / (float)RAND_MAX) - 0.5;
        }
    }

    // Initialize Layer 2 Biases
    for (int i = 0; i < num_rows_w2; ++i) {
        b2[i] = (rand() / (float)RAND_MAX) - 0.5;
    }
}

////////////////////////////////////////////////////////////////////////////////////
// Activation Functions
void ReLU(float** Z, float** result, int rows, int cols) {
    // Loop through each row
    for (int i = 0; i < rows; i++) {
        // Loop through each column
        for (int j = 0; j < cols; j++) {
            // If the current value is less than zero set to zero
            if (Z[i][j] < 0) {
                result[i][j] = 0;
            // Otherwise keep the value the same
            } else {
                result[i][j] = Z[i][j];
            }
        }
    }
}

void ReLU_deriv(float** Z, float** result, int rows, int cols) {
    // Loop through each row
    for (int i = 0; i < rows; i++) {
        // Loop through each column
        for (int j = 0; j < cols; j++) {
            // If the current value is greater than zero set to one
            if (Z[i][j] > 0) {
                result[i][j] = 1;
            // Otherwise set to zero
            } else {
                result[i][j] = 0;
            }
        }
    }
}

void softmax(float** Z, float** result, int rows, int cols) {
    // Loop through each row
    for (int i = 0; i < rows; i++) {
        // Calculate the sum of all the exponentials
	// in the current row
        float sum = 0;
        for (int j = 0; j < cols; j++) {
            result[i][j] = exp(Z[i][j]);
            sum += result[i][j];
        }
        // Loop trough each columns
        for (int j = 0; j < cols; j++) {
            // Divide each result in the current row by the sum
            result[i][j] /= sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// Forward and Backward Propagation Functions
void forward_prop(float** W1, float* b1, float** W2, float* b2, float** X,
                  float** Z1, float** A1, float** Z2, float** A2,
                  int W1_rows, int W1_cols, int W2_rows, int W2_cols, int X_cols) {
    // Preview the first image
    printf("Original Image:\n");
    preview_image(X, 0, 28, 28);

    // Multiply W1 and X
    matrix_multiply(W1, X, Z1, W1_rows, W1_cols, X_cols);

}

////////////////////////////////////////////////////////////////////////////////////
// Utility Functions
float* allocate_1d_array(int size) {
    return (float*)malloc(size * sizeof(float));
}

float** allocate_2d_array(int rows, int cols) {
    float** array = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; ++i) {
        array[i] = (float*)malloc(cols * sizeof(float));
    }
    return array;
}

void free_1d_array(float* array) {
    free(array);
}

void free_2d_array(float** array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

float** extract_first_image(float** X_train, int cols) {
    float** first_image = allocate_2d_array(1, cols);
    for (int j = 0; j < cols; ++j) {
        first_image[0][j] = X_train[0][j];
    }
    return first_image;
}

int main() {
////////////////////////////////////////////////////////////////////////////////////
// Load Data
    // Load X_train
    const int X_train_rows = 60000;
    const int X_train_cols = 784;
    float** X_train = read_csv("X_train.csv", X_train_rows, X_train_cols);

    // Load Y_train
    const int Y_train_rows = 60000;
    const int Y_train_cols = 10;
    float** Y_train = read_csv("Y_train.csv", Y_train_rows, Y_train_cols);

    // Load X_test
    const int X_test_rows = 10000;
    const int X_test_cols = 784;
    float** X_test = read_csv("X_test.csv", X_test_rows, X_test_cols);

    // Load Y_test
    const int Y_test_rows = 10000;
    const int Y_test_cols = 10;
    float** Y_test = read_csv("Y_test.csv", Y_test_rows, Y_test_cols);

    preview_data_2D(X_train, 10, 10, 0);
    preview_data_2D(Y_train, 10, 10, 0);

////////////////////////////////////////////////////////////////////////////////////
// Preview the first image
    preview_image(X_train, 0, 28, 28);

////////////////////////////////////////////////////////////////////////////////////
// Test Initialization of Parameters Function
    // Define the dimensions of the weight and biases matrices
    int num_rows_w1 = 10;
    int num_cols_w1 = 784;
    int num_rows_w2 = 10;
    int num_cols_w2 = 10;

    float** W1 = allocate_2d_array(num_rows_w1, num_cols_w1);
    float* b1 = allocate_1d_array(num_rows_w1);
    float** W2 = allocate_2d_array(num_rows_w2, num_cols_w2);
    float* b2 = allocate_1d_array(num_rows_w2);

    init_params(num_rows_w1, num_cols_w1, num_rows_w2, num_cols_w2, W1, b1, W2, b2);

    printf("Testing Parameter Initializations:\n");
    printf("W1:\n");
    preview_data_2D(W1, 3, 3, 4);
    printf("b1:\n");
    preview_data_1D(b1, 3, 4);
    printf("W2:\n");
    preview_data_2D(W2, 3, 3, 4);
    printf("b2:\n");
    preview_data_1D(b2, 3, 4);

////////////////////////////////////////////////////////////////////////////////////
// Test Activation Functions
    int rows = 2;
    int cols = 3;

    float** Z = allocate_2d_array(rows, cols);
    float** result = allocate_2d_array(rows, cols);

    // Assign some values to Z
    Z[0][0] = -1.0; Z[0][1] = 2.0; Z[0][2] = -3.0;
    Z[1][0] = 4.0; Z[1][1] = -5.0; Z[1][2] = 6.0;

    printf("Input:\n");
    preview_data_2D(Z, rows, cols, 4);

    // Testing ReLU function
    printf("\nTesting ReLU function:\n");
    ReLU(Z, result, rows, cols);
    preview_data_2D(result, rows, cols, 4);

    // Testing ReLU_deriv function
    printf("\nTesting ReLU_deriv function:\n");
    ReLU_deriv(Z, result, rows, cols);
    preview_data_2D(result, rows, cols, 4);

    // Testing softmax function
    printf("\nTesting softmax function:\n");
    softmax(Z, result, rows, cols);
    preview_data_2D(result, rows, cols, 4);

////////////////////////////////////////////////////////////////////////////////////
// Test out Linear Algebra Functions
    // Allocate and define test matrices and vector for matrix_multiply and add_vector_to_matrix
    float** A = allocate_2d_array(2, 3); // 2x3 matrix
    float** B = allocate_2d_array(3, 2); // 3x2 matrix
    float** C = allocate_2d_array(2, 2); // Result of AxB (2x2 matrix)

    float* vec = (float*)malloc(2 * sizeof(float)); // 2x1 vector

    // Assign some test values to matrices and vector
    A[0][0] = 1; A[0][1] = 2; A[0][2] = 3;
    A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;
    // Preview A
    printf("A:\n");
    preview_data_2D(A, 2, 3, 0);

    B[0][0] = 7; B[0][1] = 8;
    B[1][0] = 9; B[1][1] = 10;
    B[2][0] = 11; B[2][1] = 12;
    // Preview B
    printf("B:\n");
    preview_data_2D(B, 3, 2, 0);

    vec[0] = 1;
    vec[1] = -1;
    // Preview vec
    printf("vec:\n");
    preview_data_1D(vec, 2, 0);

    printf("\nTesting matrix_multiply function:\n");
    matrix_multiply(A, B, C, 2, 3, 2);
    preview_data_2D(C, 2, 2, 0);

    printf("\nTesting add_vector_to_matrix function:\n");
    add_vector_to_matrix(C, vec, 2, 2);
    preview_data_2D(C, 2, 2, 0);

////////////////////////////////////////////////////////////////////////////////////
// Test out Forward Propagation Function
    float** Z1 = allocate_2d_array(num_rows_w1, X_train_cols);
    float** A1 = allocate_2d_array(num_rows_w1, X_train_cols);
    float** Z2 = allocate_2d_array(num_rows_w2, X_train_cols);
    float** A2 = allocate_2d_array(num_rows_w2, X_train_cols);
    
    // Extract the first image and create a new 2D array for it
    float** X_first_image = extract_first_image(X_train, X_train_cols);


    printf("\nTesting forward_prop:\n");
    forward_prop(W1, b1, W2, b2, X_first_image, Z1, A1, Z2, A2,
		 num_rows_w1, num_cols_w1, num_rows_w2, num_cols_w2, X_train_cols);

////////////////////////////////////////////////////////////////////////////////////
// Free Memory
    free_2d_array(X_train, X_train_rows);
    free_2d_array(Y_train, Y_train_rows);
    free_2d_array(X_test, X_test_rows);
    free_2d_array(Y_test, Y_test_rows);

    free_2d_array(W1, num_rows_w1);
    free_1d_array(b1);
    free_2d_array(W2, num_rows_w2);
    free_1d_array(b2);

    free_2d_array(Z, rows);
    free_2d_array(result, rows);

    free_2d_array(A, 2);
    free_2d_array(B, 3);
    free_2d_array(C, 2);
    free(vec);

    free_2d_array(Z1, num_rows_w1);
    free_2d_array(A1, num_rows_w1);
    free_2d_array(Z2, num_rows_w2);
    free_2d_array(A2, num_rows_w2);

    free_2d_array(X_first_image, 1);

    return 0;
}
