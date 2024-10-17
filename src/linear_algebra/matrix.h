/**
 * @file matrix.h
 * @brief Defines the Matrix class for GPU-accelerated matrix operations.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stdexcept>
#include <string>

#include "vector.h"

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
     * @brief Copy constructor
     * @param other The matrix to copy from
     */
    Matrix(const Matrix& other);

    /**
     * @brief Move constructor
     * @param other The matrix to move from
     */
    Matrix(Matrix&& other) noexcept;

    /**
     * @brief Copy assignment operator
     * @param other The matrix to copy from
     * @return Reference to this matrix
     */
    Matrix& operator=(const Matrix& other);

    /**
     * @brief Move assignment operator
     * @param other The matrix to move from
     * @return Reference to this matrix
     */
    Matrix& operator=(Matrix&& other) noexcept;

    /**
     * @brief Destroy the Matrix object
     */
    ~Matrix();

    /**
     * @brief Initialize the matrix (typically sets all elements to zero)
     */
    void initialize();

    /**
     * @brief Randomize the matrix elements with values between -0.5 and 0.5
     */
    void randomize();

    /**
     * @brief Print the matrix contents
     * @param decimals Number of decimal places to display
     */
    void print(int decimals);

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

    /**
     * @brief Read data from a CSV file into the matrix
     * @param filename Path to the CSV file
     */
    void read_csv(const char* filename);

    /**
     * @brief Read a subset of data from a CSV file into the matrix
     * @param filename Path to the CSV file
     * @param startRow Starting row to read from the CSV file (0-based index)
     * @param endRow Ending row to read from the CSV file (exclusive)
     * @param fileRows Total number of rows in the CSV file
     * @param fileCols Total number of columns in the CSV file
     */
    void read_csv_limited(const char* filename,
                          int startRow,
                          int endRow,
                          int fileRows,
                          int fileCols);

    /**
     * @brief Preview a single image from the matrix
     * @param row_index Index of the row containing the image data
     * @param image_size_x Number of rows in the image
     * @param image_size_y Number of columns in the image
     */
    void preview_image(int row_index, int image_size_x, int image_size_y) const;

    /**
     * @brief Applies the ReLU activation function to the matrix.
     * @return A new Matrix object with ReLU applied.
     */
    Matrix relu() const;

    /**
     * @brief Applies the derivative of the ReLU activation function to the matrix.
     * @return A new Matrix object with ReLU derivative applied.
     */
    Matrix relu_derivative() const;

    /**
     * @brief Applies the sigmoid activation function to the matrix.
     * @return A new Matrix object with sigmoid applied.
     */
    Matrix sigmoid() const;
    
    /**
     * @brief Applies the derivative of the sigmoid activation function to the matrix.
     * @return A new Matrix object with sigmoid derivative applied.
     */
    Matrix sigmoid_derivative() const;

    /**
     * @brief Applies the softmax function to the matrix column-wise.
     * @return A new Matrix object with softmax applied.
     */
    Matrix softmax() const;

    /**
     * @brief Creates a deep copy of the matrix.
     * @return A new Matrix object with the same content as the original.
     */
    Matrix copy() const;

    /**
     * @brief Multiplies this matrix with another matrix.
     * @param other The matrix to multiply with.
     * @return A new Matrix object containing the result of the multiplication.
     * @throws std::invalid_argument if matrix dimensions are incompatible for multiplication.
     */
    Matrix multiply(const Matrix& other) const;

    /**
     * @brief Performs element-wise multiplication with another matrix.
     * @param other The matrix to multiply element-wise with.
     * @return A new Matrix object containing the result of the element-wise multiplication.
     * @throws std::invalid_argument if matrix dimensions are not identical.
     */
    Matrix multiply_elementwise(const Matrix& other) const;

    /**
     * @brief Adds a vector to each column of the matrix.
     * @param v The vector to add.
     * @throws std::invalid_argument if vector dimension doesn't match matrix rows.
     */
    void add_vector(const Vector& v);

    /**
     * @brief Subtracts another matrix from this matrix.
     * @param other The matrix to subtract.
     * @return A new Matrix object containing the result of the subtraction.
     * @throws std::invalid_argument if matrix dimensions are not identical.
     */
    Matrix subtract(const Matrix& other) const;

    /**
     * @brief Sums all elements in the matrix.
     * @return The sum of all elements in the matrix.
     */
    double sum() const;

    /**
     * @brief Divides all elements in the matrix by a scalar.
     * @param scalar The scalar to divide by.
     * @throws std::invalid_argument if scalar is zero.
     */
    void divide_scalar(double scalar);

    /**
     * @brief Multiplies all elements in the matrix by a scalar.
     * @param scalar The scalar to multiply by.
     */
    void multiply_scalar(double scalar);

    /**
     * @brief Computes the argmax of each column in the matrix.
     * @return A Vector containing the row indices of the maximum values for each column.
     */
    Vector argmax() const;

    /**
     * @brief Transposes the matrix and returns a new Matrix object.
     * @return A new Matrix object containing the transposed data.
     */
    Matrix transpose() const;

    /**
     * @brief Selects a subset of the matrix based on specified row and column ranges.
     * @param start_row Starting row index (inclusive).
     * @param end_row Ending row index (exclusive).
     * @param start_col Starting column index (inclusive).
     * @param end_col Ending column index (exclusive).
     * @return A new Matrix object containing the selected subset.
     * @throws std::out_of_range if the specified ranges are invalid.
     */
    Matrix select_batch(int start_row, int end_row, int start_col, int end_col) const;

private:
    int rows;    ///< Number of rows in the matrix
    int cols;    ///< Number of columns in the matrix
    double* d_data;  ///< Device data pointer
};

#endif // MATRIX_H
