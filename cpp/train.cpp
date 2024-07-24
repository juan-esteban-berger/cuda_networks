#include "linear_algebra.h"
#include <iostream>

int main() {
    int X_train_rows = 100;
    int X_train_cols = 10;
    int Y_train_rows = 100;
    int Y_train_cols = 1;

    // Create matrices for X_train and Y_train
    Matrix X_train(X_train_rows, X_train_cols);
    Matrix Y_train(Y_train_rows, Y_train_cols);

    // Read data from CSV files
    read_csv("data/X_train.csv", &X_train);
    read_csv("data/Y_train.csv", &Y_train);

    // Use the preview_matrix function to display X_train and Y_train
    std::cout << "Preview of X_train:" << std::endl;
    preview_matrix(&X_train, 2); // Preview with 2 decimal places

    std::cout << "Preview of Y_train:" << std::endl;
    preview_matrix(&Y_train, 2); // Preview with 2 decimal places

    // Additional random vector and matrix
    Vector additional_vector(10); // Example size
    Matrix additional_matrix(10, 10); // Example size

    // Initialize with random values
    random_vector(&additional_vector);
    random_matrix(&additional_matrix);

    // Preview the additional vector and matrix
    std::cout << "Preview of Additional Vector:" << std::endl;
    preview_vector(&additional_vector, 2);
    std::cout << "Preview of Additional Matrix:" << std::endl;
    preview_matrix(&additional_matrix, 2);

    return 0;
}
