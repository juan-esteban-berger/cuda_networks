#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

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

////////////////////////////////////////////////////////////////////////
// Function prototypes for linear_algebra.c
void read_csv(const char* filename, Matrix* matrix);
void preview_vector(Vector* v, int decimals);
void preview_matrix(Matrix* m, int decimals);
void preview_image(Matrix* m, int row_index,
		int image_size_x, int image_size_y);
void initialize_vector(Vector* v, int rows);
void initialize_matrix(Matrix* m, int rows, int cols);
void random_vector(Vector* v);
void random_matrix(Matrix* m);
void free_vector(Vector* v);
void free_matrix(Matrix* m);
void normalize_vector(Vector* v, float min, float max);
void normalize_matrix(Matrix* m, float min, float max);
void denormalize_vector(Vector* v, float min, float max);
void denormalize_matrix(Matrix* m, float min, float max);
void transpose_matrix(Matrix* original, Matrix* transpose);
void matrix_multiply(Matrix* m1, Matrix* m2, Matrix* result);
void matrix_multiply_elementwise(Matrix* m1, Matrix* m2, Matrix* result);
void matrix_subtract(Matrix* m1, Matrix* m2, Matrix* result);
void divide_matrix_by_scalar(Matrix* m, float scalar);
void sum_matrix(Matrix* m, float* result);
void add_vector_to_matrix(Matrix* m, Vector* v);
void argmax(Matrix* m, Vector* v);

#endif // LINEAR_ALGEBRA_H
