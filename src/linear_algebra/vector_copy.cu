/**
 * @file vector_copy.cu
 * @brief Implementation of the copy method for vectors.
 */

#include "vector.h"
#include <cuda_runtime.h>

/**
 * @brief Creates a deep copy of the vector.
 * @return A new Vector object with the same content as the original.
 */
Vector Vector::copy() const {
    // Create a new vector with the same number of rows
    Vector result(rows);
    
    // Copy the data from the current vector to the new vector
    cudaMemcpy(result.d_data, d_data, rows * sizeof(double), cudaMemcpyDeviceToDevice);
    
    // Return the new vector
    return result;
}
