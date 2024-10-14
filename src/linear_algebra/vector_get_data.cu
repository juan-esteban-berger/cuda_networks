/**
 * @file vector_get_data.cu
 * @brief Implementation of the Vector::get_data method.
 */
#include "vector.h"

double* Vector::get_data() const {
    // Return the pointer to the GPU memory
    return d_data;
}
