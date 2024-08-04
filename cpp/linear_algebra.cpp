#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iomanip>
#include <thread>

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

// Matrix slicing
// Matrix Matrix::iloc(int row_start, int row_end,
//                     int col_start, int col_end) {
//     Matrix result(row_end - row_start, col_end - col_start);
// 
//     for (int i = 0; i < result.rows; ++i) {
//         for (int j = 0; j < result.cols; ++j) {
//             result.data[i][j] = data[i + row_start][j + col_start];
//         }
//     }
// 
//     return result;
// }

Matrix Matrix::iloc(int row_start, int row_end, int col_start, int col_end) {
    Matrix result(row_end - row_start, col_end - col_start);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

    // Calculate rows per thread
    int rows_per_thread = result.rows / num_threads;

    // Create threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? result.rows : (t + 1) * rows_per_thread;

        threads.emplace_back([this, &result, row_start, col_start, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < result.cols; ++j) {
                    result.data[i][j] = data[i + row_start][j + col_start];
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

void Matrix::slice(int row_start, int row_end, int col_start, int col_end, Matrix& result) {
    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

    // Calculate rows per thread
    int rows_per_thread = (row_end - row_start) / num_threads;

    // Create threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = row_start + t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? row_end : (row_start + (t + 1) * rows_per_thread);

        threads.emplace_back([this, &result, row_start, col_start, start_row, end_row, col_end]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = col_start; j < col_end; ++j) {
                    result.data[i - row_start][j - col_start] = data[i][j];
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
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
// Matrix and Vector Operations
// Vector Assignment operator
// Vector& Vector::operator=(Vector& other) {
//     if (this != &other) {
//         delete[] data;
//         rows = other.rows;
//         data = new double[rows];
//         for (int i = 0; i < rows; i++) {
//             data[i] = other.data[i];
//         }
//     }
//     return *this;
// }
Vector& Vector::operator=(Vector& other) {
    if (this != &other) {
        delete[] data;
        rows = other.rows;
        data = new double[rows];

        // Determine number of threads
        unsigned int num_threads = std::thread::hardware_concurrency();
        num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

        // Calculate elements per thread
        int elements_per_thread = rows / num_threads;

        // Create threads
        std::vector<std::thread> threads;
        for (unsigned int t = 0; t < num_threads; ++t) {
            int start_index = t * elements_per_thread;
            int end_index = (t == num_threads - 1) ? rows : (t + 1) * elements_per_thread;

            threads.emplace_back([this, &other, start_index, end_index]() {
                for (int i = start_index; i < end_index; i++) {
                    data[i] = other.data[i];
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }
    return *this;
}

// // Matrix Assignment operator
// Matrix& Matrix::operator=(Matrix& other) {
//     // Copy dimensions
//     rows = other.rows;
//     cols = other.cols;
// 
//     // Allocate new memory
//     data = new double*[rows];
//     for (int i = 0; i < rows; i++) {
//         data[i] = new double[cols];
//         // Copy data
//         for (int j = 0; j < cols; j++) {
//             data[i][j] = other.data[i][j];
//         }
//     }
//     return *this;
// }
Matrix& Matrix::operator=(Matrix& other) {
    if (this != &other) {
        // Copy dimensions
        rows = other.rows;
        cols = other.cols;

        // Allocate new memory
        data = new double*[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = new double[cols];
        }

        // Determine number of threads
        unsigned int num_threads = std::thread::hardware_concurrency();
        num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

        // Calculate rows per thread
        int rows_per_thread = rows / num_threads;

        // Create threads
        std::vector<std::thread> threads;
        for (unsigned int t = 0; t < num_threads; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? rows : (t + 1) * rows_per_thread;

            threads.emplace_back([this, &other, start_row, end_row]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < cols; j++) {
                        data[i][j] = other.data[i][j];
                    }
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }
    return *this;
}

// Element-wise subtraction
// Matrix operator-(Matrix& m1, Matrix& m2) {
//     // Create matrix with m1.rows and m1.cols
//     Matrix result(m1.rows, m1.cols);
//     // Iterate over each row of the matrices
//     for (int i = 0; i < m1.rows; ++i) {
//         // Iterate over each column of the matrices
//         for (int j = 0; j < m1.cols; ++j) {
//             // Multiply corresponding elements and store in the result matrix
//             result.setValue(i, j, m1.getValues(i, j) - m2.getValues(i, j));
//         }
//     }
//     // Return the resulting matrix
//     return result;
// }
Matrix operator-(Matrix& m1, Matrix& m2) {
    // Create matrix with m1.rows and m1.cols
    Matrix result(m1.rows, m1.cols);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

    // Calculate rows per thread
    int rows_per_thread = m1.rows / num_threads;

    // Create threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? m1.rows : (t + 1) * rows_per_thread;

        threads.emplace_back([&m1, &m2, &result, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < m1.cols; ++j) {
                    result.setValue(i, j, m1.getValues(i, j) - m2.getValues(i, j));
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Return the resulting matrix
    return result;
}

// Element-wise multiplication
// Matrix operator*(Matrix& m1, Matrix& m2) {
//     // Create matrix with m1.rows and m1.cols
//     Matrix result(m1.rows, m1.cols);
//     // Iterate over each row of the matrices
//     for (int i = 0; i < m1.rows; ++i) {
//         // Iterate over each column of the matrices
//         for (int j = 0; j < m1.cols; ++j) {
//             // Multiply corresponding elements and store in the result matrix
//             result.setValue(i, j, m1.getValues(i, j) * m2.getValues(i, j));
//         }
//     }
//     // Return the resulting matrix
//     return result;
// }

Matrix operator*(Matrix& m1, Matrix& m2) {
    // Create matrix with m1.rows and m1.cols
    Matrix result(m1.rows, m1.cols);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

    // Calculate rows per thread
    int rows_per_thread = m1.rows / num_threads;

    // Create threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? m1.rows : (t + 1) * rows_per_thread;

        threads.emplace_back([&m1, &m2, &result, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < m1.cols; ++j) {
                    result.setValue(i, j, m1.getValues(i, j) * m2.getValues(i, j));
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Return the resulting matrix
    return result;
}

// Compute portion of matrix multiplication
void compute_portion(Matrix& result, Matrix& m1, Matrix& m2,
                     int start_row, int end_row) {
    // Iterate over the assigned rows
    for (int i = start_row; i < end_row; ++i) {
        // Iterate over the columns of m1 (=rows of m2)
        for (int k = 0; k < m1.cols; ++k) {
            // Get the value of m1 at row i and column k
            const double m1_ik = m1.getValues(i, k);
            
            // Iterate over the columns of m2
            for (int j = 0; j < m2.cols; ++j) {
                // Calculate dot product
                result.data[i][j] += m1_ik * m2.getValues(k, j);
            }
        }
    }
}

// Matrix Multiplication
Matrix matmul(Matrix& m1, Matrix& m2) {
    // Create result matrix
    Matrix result(m1.rows, m2.cols);

    // Get number of concurrent threads supported by the hardware
    unsigned int num_threads = std::thread::hardware_concurrency();
    
    // Limit threads between 1 and 16, if fails default to 4
    if (num_threads == 0 || num_threads > 16) {
        num_threads = 4;
    }

    // Calculate the number of rows each thread should process
    int rows_per_thread = m1.rows / num_threads;

    // Create a vector to store the thread objects
    std::vector<std::thread> threads;

    // Create and launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        // Calculate the starting row for this thread
        int start_row = t * rows_per_thread;
        
        // Initialize int for ending row
        int end_row;
        // If this is the last thread
        if (t == num_threads - 1) {
            // Assign the remaining rows to this thread
            end_row = m1.rows;
        // If not the last thread
        } else {
            // Assign the next set of rows to this thread
            end_row = (t + 1) * rows_per_thread;
        }
        
        // Create and launch a new thread
        threads.emplace_back(
            // Function to be executed by the new thread
            compute_portion,
            
            // std::ref creates a reference wrapper for 'result'
            // ensuring thread works on original matrix, not a copy
            std::ref(result),
            
            // std::ref creates a reference wrapper for 'm1'
            // ensuring thread works on original matrix, not a copy
            std::ref(m1),
            
            // std::ref creates a reference wrapper for 'm2'
            // ensuring thread works on original matrix, not a copy
            std::ref(m2),
            
            // start_row
            start_row,
            
            // end_row
            end_row);
    }

    // Wait for all threads to complete their computations
    for (auto& thread : threads) {
        // Join each thread, blocking until it finishes
        thread.join();
    }

    // Return the computed result matrix
    return result;
}

// Matrix-vector addition (unchanged from previous example)
// Matrix operator+(Matrix& m, Vector& v) {
//     // Initialize matrix
//     Matrix result(m.rows, m.cols);
//     // Iterate over each row of the matrix
//     for (int i = 0; i < m.rows; ++i) {
//         // Iterate over each column of the matrix
//         for (int j = 0; j < m.cols; ++j) {
//             // Add the corresponding vector element to each matrix element
//             result.setValue(i, j, m.getValues(i, j) + v.getValues(i));
//         }
//     }
//     // Return the resulting matrix
//     return result;
// }

Matrix operator+(Matrix& m, Vector& v) {
    // Initialize matrix
    Matrix result(m.rows, m.cols);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

    // Calculate rows per thread
    int rows_per_thread = m.rows / num_threads;

    // Create threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? m.rows : (t + 1) * rows_per_thread;

        threads.emplace_back([&m, &v, &result, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < m.cols; ++j) {
                    result.setValue(i, j, m.getValues(i, j) + v.getValues(i));
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Return the resulting matrix
    return result;
}

// Matrix-scalar division
// Matrix operator/(Matrix& m, double scalar) {
//     // Initialize matrix with m.rows and m.cols
//     Matrix result(m.rows, m.cols);
//     // Iterate over each row of the matrix
//     for (int i = 0; i < m.rows; ++i) {
//         // Iterate over each column of the matrix
//         for (int j = 0; j < m.cols; ++j) {
//             // Divide each element by the scalar
//             result.setValue(i, j, m.getValues(i, j) / scalar);
//         }
//     }
//     // Return the resulting matrix
//     return result;
// }
Matrix operator/(Matrix& m, double scalar) {
    // Initialize matrix with m.rows and m.cols
    Matrix result(m.rows, m.cols);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

    // Calculate rows per thread
    int rows_per_thread = m.rows / num_threads;

    // Create threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? m.rows : (t + 1) * rows_per_thread;

        threads.emplace_back([&m, &result, scalar, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < m.cols; ++j) {
                    result.setValue(i, j, m.getValues(i, j) / scalar);
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Return the resulting matrix
    return result;
}

// Vector-scalar division
// Vector operator/(Vector& v, double scalar) {
//     // Initialize vector with v.rows
//     Vector result(v.rows);
//     // Iterate over each row of the vector
//     for (int i = 0; i < v.rows; i++) {
//         // Divide each element
//         result.setValue(i, v.getValues(i) / scalar);
//     }
//     // Return the resulting vector
//     return result;
// }

Vector operator/(Vector& v, double scalar) {
    // Initialize vector with v.rows
    Vector result(v.rows);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

    // Calculate elements per thread
    int elements_per_thread = v.rows / num_threads;

    // Create threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_index = t * elements_per_thread;
        int end_index = (t == num_threads - 1) ? v.rows : (t + 1) * elements_per_thread;

        threads.emplace_back([&v, &result, scalar, start_index, end_index]() {
            for (int i = start_index; i < end_index; i++) {
                result.setValue(i, v.getValues(i) / scalar);
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Return the resulting vector
    return result;
}

// Sum matrix columns
// Vector sum_columns(Matrix& m) {
//     // Initialize vector with m.rows
//     Vector result(m.rows);
//     // Iterate over each row of the matrix
//     for (int i = 0; i < m.rows; ++i) {
//         // Initialize sum to 0
//         double sum = 0;
//         // Iterate over each column of the matrix
//         for (int j = 0; j < m.cols; ++j) {
//             // Add the element to the sum
//             sum += m.getValues(i, j);
//         }
//         // Store the sum in the result vector
//         result.setValue(i, sum);
//     }
//     // Return the result vector
//     return result;
// }
Vector sum_columns(Matrix& m) {
    // Initialize vector with m.rows
    Vector result(m.rows);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0 || num_threads > 16) ? 4 : num_threads;

    // Calculate rows per thread
    int rows_per_thread = m.rows / num_threads;

    // Create threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? m.rows : (t + 1) * rows_per_thread;

        threads.emplace_back([&m, &result, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                double sum = 0;
                for (int j = 0; j < m.cols; ++j) {
                    sum += m.getValues(i, j);
                }
                result.setValue(i, sum);
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Return the result vector
    return result;
}

// Argmax of each column
Vector argmax(Matrix& m) {
    // Initialize vector with m.cols
    Vector result(m.cols);
    // Iterate over each column of the matrix
    for (int j = 0; j < m.cols; ++j) {
        int maxIndex = 0;
        double maxValue = m.getValues(0, j);
        // Iterate over each row of the matrix
        for (int i = 1; i < m.rows; ++i) {
            // Check if element is greater than maxValue
            if (m.getValues(i, j) > maxValue) {
                maxValue = m.getValues(i, j);
                maxIndex = i;
            }
        }
        // Set Index
        result.setValue(j, maxIndex);
    }
    return result;
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
                    // Jump to last 5 columns
                    k = m->cols - 5;
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
