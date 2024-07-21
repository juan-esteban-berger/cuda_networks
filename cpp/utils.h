#ifndef UTILS_H
#define UTILS_H

#include <string>

////////////////////////////////////////////////////////////////////
// Series Class
class Series {
private:
    double* values;
    int length;

public:
    Series(int length);
    void setValues(double* values);
    double* getValues();
    int getLength();
    void randomize();
    void print(int decimals);
    void normalize(double min, double max);
    void denormalize(double min, double max);
    ~Series();
};

////////////////////////////////////////////////////////////////////
// DataFrame Class
class DataFrame {
private:
    double** values;
    int numRows, numCols;

public:
    DataFrame(int numRows, int numCols);
    void setValues(double** values);
    double** getValues();
    int getNumRows();
    int getNumCols();
    void randomize();
    void read_csv(const std::string &filename);
    void read_csv_limited(const std::string &filename,
                          int startRow, int endRow);
    void print(int decimals);
    DataFrame transpose();
    void normalize(double min, double max);
    void denormalize(double min, double max);
    ~DataFrame();
};

#endif
