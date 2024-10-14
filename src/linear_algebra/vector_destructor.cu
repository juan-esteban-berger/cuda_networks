/**
 * @file vector_destructor.cu
 * @brief Implementation of the Vector class destructor.
 */
#include "vector.h"
#include <cuda_runtime.h>

Vector::~Vector() {
    // Free the GPU memory allocated for this vector
    cudaFree(d_data);
}
