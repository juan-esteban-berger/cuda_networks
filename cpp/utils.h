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
    void read_csv(const std::string &filename);
    void print(int decimals);
    ~DataFrame();
};

#endif
