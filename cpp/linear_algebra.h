#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

//////////////////////////////////////////////////////////////////
// Vector Class
class Vector {
public:
    double* data;
    int rows;

    Vector(int r);
    ~Vector();
    void setValue(int index, double value);
    double getValues(int index);
};

//////////////////////////////////////////////////////////////////
// Matrix Class
class Matrix {
public:
    double** data;
    int rows;
    int cols;

    Matrix(int r, int c);
    ~Matrix();
    void setValue(int row, int col, double value);
    double getValues(int row, int col);
};

//////////////////////////////////////////////////////////////////
// Read from CSV
void read_csv(const char* filename, Matrix* matrix);
void read_csv_limited(const char* filename, Matrix* matrix_subset, int startRow, int endRow, int fileRows, int fileCols);

/////////////////////////////////////////////////////////////////
// Preview Functions
void preview_matrix(Matrix* m, int decimals);
void preview_vector(Vector* v, int decimals);

//////////////////////////////////////////////////////////////////
// Randomize Functions
void random_vector(Vector* v);
void random_matrix(Matrix* m);

//////////////////////////////////////////////////////////////////
// Transpose Function
Matrix* transpose_matrix(Matrix* m);

//////////////////////////////////////////////////////////////////
// Normalization Functions
void normalize_vector(Vector* v, double min, double max);
void normalize_matrix(Matrix* m, double min, double max);
void denormalize_vector(Vector* v, double min, double max);
void denormalize_matrix(Matrix* m, double min, double max);

#endif // LINEAR_ALGEBRA_H