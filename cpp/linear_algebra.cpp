#include <iostream>
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iomanip>

#include "linear_algebra.h"

//////////////////////////////////////////////////////////////////
// Vector Class
Vector::Vector(int r) : rows(r) {
    data = new double[rows];
    memset(data, 0, rows * sizeof(double));
}

Vector::~Vector() {
    delete[] data;
}

void Vector::setValue(int index, double value) {
    data[index] = value;
}

double Vector::getValues(int index) {
    return data[index];
}

//////////////////////////////////////////////////////////////////
// Matrix Class
Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        data[i] = new double[cols];
        memset(data[i], 0, cols * sizeof(double));
    }
}

Matrix::~Matrix() {
    for (int i = 0; i < rows; ++i) {
        delete[] data[i];
    }
    delete[] data;
}

void Matrix::setValue(int row, int col, double value) {
    data[row][col] = value;
}

double Matrix::getValues(int row, int col) {
    return data[row][col];
}

//////////////////////////////////////////////////////////////////
// Read from CSV
void read_csv(const char* filename, Matrix* matrix) {
    // Open the file with the filename provided.
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Iterate over the number of rows
    for (int row = 0; row < matrix->rows; ++row) {
        // Iterate over the number of columns
        for (int col = 0; col < matrix->cols; ++col) {
            // Reads a double value from the file and stores
            // it in the matrix using setValue method.
            fscanf(file, "%lf,", &matrix->data[row][col]);
        }
    }

    // Close the file
    fclose(file);
}

void read_csv_limited(const char* filename, Matrix* matrix_subset,
                      int startRow, int endRow, int fileRows, int fileCols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Iterate over the total rows in the file
    for (int row = 0; row < fileRows; ++row) {
        // Iterate over the columns in the file
        for (int col = 0; col < fileCols; ++col) {
            if (row >= startRow && row < endRow) {
                // Reads a double value from the file
                // and stores it in the matrix.
                fscanf(file, "%lf,", &matrix_subset->data[row - startRow][col]);
            } else {
                // Skip the values outside range
                double temp;
                fscanf(file, "%lf,", &temp);
            }
        }
    }

    // Close the file
    fclose(file);
}

//////////////////////////////////////////////////////////////////
// Preview Functions
void preview_vector(Vector* v, int decimals) {
    std::cout << "Vector with " << v->rows << " rows:\n";
    std::cout << "\t0:\t\n";

    for (int i = 0; i < v->rows; i++) {
        if (i == 5 && v->rows > 10) {
            std::cout << "...\t...\n";
            // Skip to last 5 rows
            i = v->rows - 5;
        }
        std::cout << i << ":\t" << std::fixed << std::setprecision(decimals) << v->getValues(i) << "\n";
    }
    std::cout << "\n";
}

// Function to preview matrix data
void preview_matrix(Matrix* m, int decimals) {
    std::cout << "Matrix with " << m->rows << " rows and " << m->cols << " columns:\n";
    std::cout << "\t";
    for (int j = 0; j < m->cols; j++) {
        if (j == 5 && m->cols > 10) {
            std::cout << "...\t";
            // Skip to last 5 columns
            j = m->cols - 5;
        }
        std::cout << j << ":\t";
    }
    std::cout << "\n";

    for (int i = 0; i < m->rows; i++) {
        if (i == 5 && m->rows > 10) {
            std::cout << "...\t";
            for (int k = 0; k < m->cols; k++) {
                if (k == 5 && m->cols > 10) {
                    std::cout << "...\t";
                    k = m->cols - 5;  // Jump to last 5 columns
                }
                std::cout << "...\t";
            }
            std::cout << "\n";
            // Skip to last 5 rows
            i = m->rows - 5;
        }
        std::cout << i << ":\t";
        for (int j = 0; j < m->cols; j++) {
            if (j == 5 && m->cols > 10) {
                std::cout << "...\t";
                j = m->cols - 5;
            }
            std::cout << std::fixed << std::setprecision(decimals) << m->getValues(i, j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

//////////////////////////////////////////////////////////////////
// Randomize Functions
void random_vector(Vector* v) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Iterate over the rows
    for (int i = 0; i < v->rows; i++) {
        // Generate random double between -0.5 and 0.5
        double r = static_cast<double>(rand()) / RAND_MAX - 0.5;

        // Store random double in vector
        v->data[i] = r;
    }
}

void random_matrix(Matrix* m) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Iterate over the rows
    for (int i = 0; i < m->rows; i++) {
        // Iterate over the columns
        for (int j = 0; j < m->cols; j++) {
            // Generate random double between -0.5 and 0.5
            double r = static_cast<double>(rand()) / RAND_MAX - 0.5;

            // Store random double in matrix
            m->data[i][j] = r;
        }
    }
}

//////////////////////////////////////////////////////////////////
// Transpose Functions
Matrix* transpose_matrix(Matrix* m) {
    Matrix* transposed = new Matrix(m->cols, m->rows);
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            transposed->setValue(j, i, m->getValues(i, j));
        }
    }
    
    return transposed;
}

//////////////////////////////////////////////////////////////////
// Normalization Functions
void normalize_vector(Vector* v, double min, double max) {
    for (int i = 0; i < v->rows; i++) {
        v->data[i] = (v->data[i] - min) / (max - min);
    }
}

void normalize_matrix(Matrix* m, double min, double max) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = (m->data[i][j] - min) / (max - min);
        }
    }
}

// Denormalizing Functions
void denormalize_vector(Vector* v, double min, double max) {
    for (int i = 0; i < v->rows; i++) {
        v->data[i] = v->data[i] * (max - min) + min;
    }
}

void denormalize_matrix(Matrix* m, double min, double max) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = m->data[i][j] * (max - min) + min;
        }
    }
}
