/**
 * @file matrix_get_rows.cu
 * @brief Implementation of the Matrix::get_rows method.
 */
#include "matrix.h"

int Matrix::get_rows() const {
    // Return the number of rows in the matrix
    return rows;
}
