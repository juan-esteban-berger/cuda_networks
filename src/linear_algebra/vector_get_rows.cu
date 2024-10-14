/**
 * @file vector_get_rows.cu
 * @brief Implementation of the Vector::get_rows method.
 */
#include "vector.h"

int Vector::get_rows() const {
    // Return the number of elements in the vector
    return rows;
}
