/**
 * @file matrix_get_cols.cu
 * @brief Implementation of the Matrix::get_cols method.
 */
#include "matrix.h"

int Matrix::get_cols() const {
    // Return the number of columns in the matrix
    return cols;
}
