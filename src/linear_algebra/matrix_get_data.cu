/**
 * @file matrix_get_data.cu
 * @brief Implementation of the Matrix::get_data method.
 */
#include "matrix.h"

double* Matrix::get_data() const {
    // Return the pointer to the GPU memory
    return d_data;
}
