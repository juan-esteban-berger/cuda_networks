#include "utils.h"

int main() {
    Series s(15);

    double values[] = {1.0, 2.0, 3.0, 4.0, 5.0,
                       6.0, 7.0, 8.0, 9.0, 10.0,
                       11.0, 12.0, 13.0, 14.0, 15.0};
    s.setValues(values);

    s.print(2);

    // DataFrame df(60000, 784);

    // df.read_csv("data/X_train.csv");

    // DataFrame df(1000, 784);

    // df.read_csv_limited("data/X_train.csv", 0, 1000);
    
    // Dimensions of the DataFrame
    int numRows = 100;
    int numCols = 30;

    // Create a DataFrame instance
    DataFrame df(numRows, numCols);

    // Create sample data to set in the DataFrame using malloc
    double** data = (double**) malloc(numCols * sizeof(double*));
    for (int i = 0; i < numCols; i++) {
        data[i] = (double*) malloc(numRows * sizeof(double));
        for (int j = 0; j < numRows; j++) {
            data[i][j] = i + 0.1 * j;
        }
    }

    // Set the values in the DataFrame
    df.setValues(data);

    df.print(2);

    return 0;
}
