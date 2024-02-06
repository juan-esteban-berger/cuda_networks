#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <stdio.h>

////////////////////////////////////////////////////////////////////////
// Structs for vectors and matrices
typedef struct {
    double* data;
    int rows;
} Vector;

typedef struct {
    double** data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    double* data;
    int rows;
    int cols;
} Matrix_GPU;

////////////////////////////////////////////////////////////////////////
// Function prototypes for cuda_linear_algebra.cu
void read_csv(const char* filename, Matrix* matrix);
void read_csv_limited(const char* filename, Matrix* matrix,
        int startRow, int endRow,
        int fileRows, int fileCols);
void preview_vector(Vector* v, int decimals);
void preview_vector_GPU(Vector* v, int decimals);
void preview_matrix(Matrix* m, int decimals);
void preview_matrix_GPU(Matrix_GPU* m, int decimals);
void preview_image(Matrix* m, int row_index,
        int image_size_x, int image_size_y);
void initialize_vector(Vector* v, int rows);
void initialize_vector_on_device(Vector* v, int rows);
void initialize_matrix(Matrix* m, int rows, int cols);
void initialize_matrix_on_device(Matrix_GPU* m, int rows, int cols);
void copy_vector_to_device(Vector* h_v, Vector* d_v);
void copy_matrix_to_device(Matrix* h_m, Matrix_GPU* d_m);
void copy_vector_to_host(Vector* h_v, Vector* d_v);
void copy_matrix_to_host(Matrix* h_m, Matrix_GPU* d_m);
void copy_matrix_range_to_matrix_GPU(Matrix_GPU* d_m, Matrix_GPU* d_m_subset,
        int startRow, int endRow);
void copy_random_matrix_range_to_matrix_GPU(Matrix_GPU* d_x, Matrix_GPU* d_x_subset,
        Matrix_GPU* d_y, Matrix_GPU* d_y_subset,
        int numRows, int totalRows);
void random_vector(Vector* v);
void random_matrix(Matrix* m);
void free_vector(Vector* v);
void free_vector_on_device(Vector* v);
void free_matrix(Matrix* m);
void free_matrix_on_device(Matrix_GPU* m);
void normalize_vector(Vector* v, double min, double max);
void normalize_matrix(Matrix* m, double min, double max);
void denormalize_vector(Vector* v, double min, double max);
void denormalize_matrix(Matrix* m, double min, double max);
__global__ void scalar_division_GPU(double *v, double scalar);
void transpose_matrix(Matrix* original, Matrix* transpose);
__global__ void transpose_matrix_GPU(double *input, double *output, int rows, int cols);
void matrix_multiply(Matrix* m1, Matrix* m2, Matrix* result);
__global__ void matrix_multiply_GPU(double *a, double *b, double *c,
				    int aRows, int aCols, int bCols);
void matrix_multiply_elementwise(Matrix* m1, Matrix* m2, Matrix* result);
__global__ void matrix_multiply_elementwise_GPU(double *a, double *b, double *c,
				    int rows, int cols);
void matrix_subtract(Matrix* m1, Matrix* m2, Matrix* result);
__global__ void matrix_subtract_GPU(double *a, double *b, double *c, int rows, int cols);
void divide_matrix_by_scalar(Matrix* m, double scalar);
__global__ void divide_matrix_by_scalar_GPU(double *a, double scalar, int rows, int cols);
void sum_matrix(Matrix* m, double* result);
__global__ void sum_matrix_GPU(double *m, double *result, int rows, int cols);
void add_vector_to_matrix(Matrix* m, Vector* v);
__global__ void add_vector_to_matrix_GPU(double *m, double *v, int rows, int cols);
void argmax(Matrix* m, Vector* v);
__global__ void argmax_GPU(double *m, double *result, int rows, int cols);

#endif // LINEAR_ALGEBRA_H
