/**
 * @file matrix.h
 * @brief Defines the Matrix class for GPU-accelerated matrix operations.
 */

#ifndef MATRIX_H
#define MATRIX_H

/**
 * @class Matrix
 * @brief Represents a matrix with GPU-accelerated operations.
 */
class Matrix {
public:
    /**
     * @brief Construct a new Matrix object
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     */
    Matrix(int rows, int cols);

    /**
     * @brief Destroy the Matrix object
     */
    ~Matrix();

    /**
     * @brief Initialize the matrix (typically sets all elements to zero)
     */
    void initialize();

    /**
     * @brief Print the matrix contents
     */
    void print();

    /**
     * @brief Get the number of rows in the matrix
     * @return int Number of rows
     */
    int get_rows() const;

    /**
     * @brief Get the number of columns in the matrix
     * @return int Number of columns
     */
    int get_cols() const;

    /**
     * @brief Get the raw data pointer of the matrix
     * @return double* Pointer to the matrix data on the device
     */
    double* get_data() const;

private:
    int rows;    ///< Number of rows in the matrix
    int cols;    ///< Number of columns in the matrix
    double* d_data;  ///< Device data pointer
};

#endif // MATRIX_H
