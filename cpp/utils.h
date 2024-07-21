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
    void print(int decimals);
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
    void read_csv(const std::string &filename);
    void read_csv_limited(const std::string &filename,
                          int startRow, int endRow);
    void print(int decimals);
    DataFrame transpose();
    ~DataFrame();
};

#endif
