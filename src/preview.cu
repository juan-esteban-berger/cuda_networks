#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "linear_algebra.h"

////////////////////////////////////////////////////////////////////////
// Main function
int main() {

////////////////////////////////////////////////////////////////////////
// Define constants for number of rows and neurons
    const int NUM_ROWS_TRAIN = 60000;
    const int NUM_NEURONS_INPUT = 784;
    const int NUM_NEURONS_OUTPUT = 10;

////////////////////////////////////////////////////////////////////////
// Load Training Data
    // Read in data from X_train.csv
    Matrix X_train;
    initialize_matrix(&X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);
    read_csv("data/X_train.csv", &X_train);

    // Read in data from Y_train.csv
    Matrix Y_train;
    initialize_matrix(&Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);
    read_csv("data/Y_train.csv", &Y_train);

////////////////////////////////////////////////////////////////////////
// Preview Data
    // Preview X_train
    printf("X_train:\n");
    preview_matrix(&X_train, 2);

    // Preview Y_train
    printf("Y_train:\n");
    preview_matrix(&Y_train, 2);

    // Preview the first image from X_train
    printf("First image from X_train:\n");
    preview_image(&X_train, 0, 28, 28);

////////////////////////////////////////////////////////////////////////
// Free memory
    // Free training data
    free_matrix(&X_train);
    free_matrix(&Y_train);

    return 0;
}

