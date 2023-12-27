#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////////////////
// Loading and Preview Functions
float* read_csv(const char* filename, int num_rows, int num_cols) {
    // Open the file with the filename provided. 'r' opens the file for reading.
    FILE* file = fopen(filename, "r");

    // Allocate memory for 1D Array
    float* data = (float*)malloc(num_rows * num_cols * sizeof(float));

    // Iterate over the number of rows
    for (int row = 0; row < num_rows; ++row) {
        // Iterate over the number of columns
        for (int col = 0; col < num_cols; ++col) {
            // Reads a float value from the file and stores
            // it in the data array at the correct position.
            fscanf(file, "%f,", &data[row * num_cols + col]);
        }
    }

    // Close the file
    fclose(file);

    // Return the pointer to the array
    return data;
}

void preview_image(float* X_train, int cols, int img_index) {
    // Adjust the starting pointer to the correct image based on the index
    X_train += img_index * cols;

    // Iterate through each pixel of the image
    for (int i = 0; i < cols; ++i) {
        // Print the pixel value
        if (X_train[i] == 0.0f) {
            printf("    ");
        } else {
            printf("%03.0f ", X_train[i]);
        }

        // Insert a newline after every 28 pixels
        // (to create a 28x28 grid for visual interpretation)
        if ((i + 1) % 28 == 0) {
            printf("\n");
        }
    }

    // Print an extra newline at the end to separate subsequent outputs
    printf("\n");
}

void print_matrix(float** mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
}

////////////////////////////////////////////////////////////////////////////////////
// Initialize Parameters
void init_params(int num_rows_wb1, int num_cols_w1, int num_rows_wb2, int num_cols_w2, float* W1, float* b1, float* W2, float* b2) {
    // Seed Random Number Generator with Current Time
    srand(time(0));

    // Initialize Layer 1 Weights
    for (int i = 0; i < num_rows_wb1*num_cols_w1; ++i) {
        W1[i] = (rand() / (float)RAND_MAX) - 0.5;
    }

    // Initialize Layer 1 Biases
    for (int i = 0; i < num_rows_wb1; ++i) {
        b1[i] = (rand() / (float)RAND_MAX) - 0.5;
    }

    // Initialize Layer 2 Weights
    for (int i = 0; i < num_rows_wb2*num_cols_w2; ++i) {
        W2[i] = (rand() / (float)RAND_MAX) - 0.5;
    }

    // Initialize Layer 2 Biases
    for (int i = 0; i < num_rows_wb2; ++i) {
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

int main() {
////////////////////////////////////////////////////////////////////////////////////
// Load Data
    // Load X_train
    const int X_train_rows = 60000;
    const int X_train_cols = 784;
    float* X_train = read_csv("X_train.csv", X_train_rows, X_train_cols);

    // Load Y_train
    const int Y_train_rows = 60000;
    const int Y_train_cols = 10;
    float* Y_train = read_csv("Y_train.csv", Y_train_rows, Y_train_cols);

    // Load X_test
    const int X_test_rows = 10000;
    const int X_test_cols = 784;
    float* X_test = read_csv("X_test.csv", X_test_rows, X_test_cols);

    // Load Y_test
    const int Y_test_rows = 10000;
    const int Y_test_cols = 10;
    float* Y_test = read_csv("Y_test.csv", Y_test_rows, Y_test_cols);

////////////////////////////////////////////////////////////////////////////////////
// Preview Data
    // Preview the first image
    int image_index = 0;
    preview_image(X_train, X_train_cols, image_index);

    // Preview the first 10 one-hot encoded labels
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < Y_train_cols; ++j) {
            printf("%d ", (int)(Y_train[i * Y_train_cols + j] + 0.5f));
        }
        printf("\n");
    }

    printf("\n");

////////////////////////////////////////////////////////////////////////////////////
// Test Initialization of Parameters Function
    // Initialize parameters
    int num_rows_wb1 = 10;
    int num_cols_w1 = 784;
    int num_rows_wb2 = 10;
    int num_cols_w2 = 10;

    float* W1 = (float *) malloc(num_rows_wb1 * num_cols_w1 * sizeof(float));
    float* b1 = (float *) malloc(num_rows_wb1 * sizeof(float));
    float* W2 = (float *) malloc(num_rows_wb2 * num_cols_w2 * sizeof(float));
    float* b2 = (float *) malloc(num_rows_wb2 * sizeof(float));

    init_params(num_rows_wb1, num_cols_w1, num_rows_wb2, num_cols_w2, W1, b1, W2, b2);

    printf("Testing Parameter Initializations:");
    // Print the first few values of each to test
    for (int i = 0; i < 3; i++) {
        printf("W1[%d]: %f\n", i, W1[i]);
        printf("b1[%d]: %f\n", i, b1[i]);
        printf("W2[%d]: %f\n", i, W2[i]);
        printf("b2[%d]: %f\n\n", i, b2[i]);
    }

////////////////////////////////////////////////////////////////////////////////////
// Test Activation Functions
    int rows = 2;
    int cols = 3;

    // Initialize a 2D array
    float **Z = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
        Z[i] = (float *)malloc(cols * sizeof(float));

    // Assign some values to Z
    Z[0][0] = -1.0; Z[0][1] = 2.0; Z[0][2] = -3.0;
    Z[1][0] = 4.0; Z[1][1] = -5.0; Z[1][2] = 6.0;

    // Initialize another 2D array for the result
    float **result = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
        result[i] = (float *)malloc(cols * sizeof(float));

    // Print Original Matrix
    printf("Input:\n");
    print_matrix(Z, rows, cols);

    // Testing ReLU function
    printf("\nTesting ReLU function:\n");
    ReLU(Z, result, rows, cols);
    print_matrix(result, rows, cols);

    // Testing ReLU_deriv function
    printf("\nTesting ReLU_deriv function:\n");
    ReLU_deriv(Z, result, rows, cols);
    print_matrix(result, rows, cols);

    // Testing softmax function
    printf("\nTesting softmax function:\n");
    softmax(Z, result, rows, cols);
    print_matrix(result, rows, cols);

    // Free memory from original matrix
    for (int i = 0; i < rows; i++)
        free(Z[i]);
    free(Z);

    // Free memory for the result matrix
    for (int i = 0; i < rows; i++)
        free(result[i]);
    free(result);

////////////////////////////////////////////////////////////////////////////////////
// Free Memory
    free(W1);
    free(b1);
    free(W2);
    free(b2);

    return 0;
}
