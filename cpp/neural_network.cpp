#include <cmath>

#include "utils.h"
#include "neural_network.h"

////////////////////////////////////////////////////////////////////
// Sigmoid Function Classes
void Sigmoid::function(DataFrame& Z) {
    int numRows = Z.getNumRows();
    int numCols = Z.getNumCols();

    // Allocate temporary array
    double** tempValues = (double**) malloc(numCols * sizeof(double*));
    for (int i = 0; i < numCols; ++i) {
        tempValues[i] = (double*) malloc(numRows * sizeof(double));
    }

    // Load data into tempValues
    double** values = Z.getValues();
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numRows; ++j) {
            tempValues[i][j] = values[i][j];
        }
    }

    // Apply sigmoid on tempValues
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numRows; ++j) {
            tempValues[i][j] = 1.0 / (1.0 + exp(-tempValues[i][j]));
        }
    }

    // Set the values back to Z
    Z.setValues(tempValues);

    // Free temporary array
    for (int i = 0; i < numCols; ++i) {
        free(tempValues[i]);
    }
    free(tempValues);
}

void Sigmoid::derivative(DataFrame& Z) {
    int numRows = Z.getNumRows();
    int numCols = Z.getNumCols();

    // Allocate temporary array
    double** tempValues = (double**) malloc(numCols * sizeof(double*));
    for (int i = 0; i < numCols; ++i) {
        tempValues[i] = (double*) malloc(numRows * sizeof(double));
    }

    // Load data into tempValues
    double** values = Z.getValues();
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numRows; ++j) {
            tempValues[i][j] = values[i][j];
        }
    }

    // Apply sigmoid derivative on tempValues
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numRows; ++j) {
            double sigmoidValue = 1.0 / (1.0 + exp(-tempValues[i][j]));
            tempValues[i][j] = sigmoidValue * (1 - sigmoidValue);
        }
    }

    // Set the values back to Z
    Z.setValues(tempValues);

    // Free temporary array
    for (int i = 0; i < numCols; ++i) {
        free(tempValues[i]);
    }
    free(tempValues);
}
