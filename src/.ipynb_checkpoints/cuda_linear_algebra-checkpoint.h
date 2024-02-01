#ifndef CUDA_LINEAR_ALGEBRA_H
#define CUDA_LINEAR_ALGEBRA_H

#include <stdio.h>

////////////////////////////////////////////////////////////////////////
// Structs for vectors and matrices
typedef struct {
    float* data;
    int rows;
} Vector;

typedef struct {
    float** data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix_GPU;

////////////////////////////////////////////////////////////////////////
// Function prototypes for cuda_linear_algebra.cu
void read_csv(const char* filename, Matrix* matrix);
void preview_vector(Vector* v, int decimals);
void preview_matrix(Matrix* m, int decimals);
void preview_image(Matrix* m, int row_index,
		int image_size_x, int image_size_y);
void initialize_vector(Vector* v, int rows);
void initialize_vector_on_device(Vector* v, int rows); // New
void initialize_matrix(Matrix* m, int rows, int cols);
void initialize_matrix_on_device(Matrix_GPU* m, int rows, int cols); // New
void copy_vector_to_device(Vector* h_v, Vector* d_v); // New
void copy_matrix_to_device(Matrix* h_m, Matrix_GPU* d_m); // New
void copy_vector_to_host(Vector* h_v, Vector* d_v); // New
void copy_matrix_to_host(Matrix* h_m, Matrix_GPU* d_m); // New
void random_vector(Vector* v);
void random_matrix(Matrix* m);
void free_vector(Vector* v);
void free_vector_on_device(Vector* v); // New
void free_matrix(Matrix* m);
void free_matrix_on_device(Matrix_GPU* m); // New
void normalize_vector(Vector* v, float min, float max);
void normalize_matrix(Matrix* m, float min, float max);
void denormalize_vector(Vector* v, float min, float max);
void denormalize_matrix(Matrix* m, float min, float max);
void transpose_matrix(Matrix* original, Matrix* transpose);
__global__ void transpose_matrix_GPU(float *input, float *output, int rows, int cols); // New
void matrix_multiply(Matrix* m1, Matrix* m2, Matrix* result);
__global__ void matrix_multiply_GPU(float *a, float *b, float *c, int aRows, int aCols, int bCols); // New
void matrix_multiply_elementwise(Matrix* m1, Matrix* m2, Matrix* result);
__global__ void matrix_multiply_elementwise_GPU(float *a, float *b, float *c, int rows, int cols); // New
void matrix_subtract(Matrix* m1, Matrix* m2, Matrix* result);
__global__ void matrix_subtract_GPU(float *a, float *b, float *c, int rows, int cols); // New
void divide_matrix_by_scalar(Matrix* m, float scalar);
__global__ void divide_matrix_by_scalar_GPU(float *a, float scalar, int rows, int cols); // New
void sum_matrix(Matrix* m, float* result);
__global__ void sum_matrix_GPU(float *m, float *result, int rows, int cols); // New
void add_vector_to_matrix(Matrix* m, Vector* v);
__global__ void add_vector_to_matrix_GPU(float *m, float *v, int rows, int cols); // New
void argmax(Matrix* m, Vector* v);
__global__ void argmax_GPU(float *m, float *result, int rows, int cols); // New

#endif // LINEAR_ALGEBRA_H
