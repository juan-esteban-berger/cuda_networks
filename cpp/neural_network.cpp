#include <cmath>

#include "utils.h"
#include "neural_network.h"

////////////////////////////////////////////////////////////////////
// Activation Function Classes
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

void Softmax::function(DataFrame& Z) {
    int numRows = Z.getNumRows();
    int numCols = Z.getNumCols();

    // Allocate temporary array
    double** tempValues = (double**) malloc(numCols * sizeof(double*));
    // Loop through each column
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each column
        tempValues[i] = (double*) malloc(numRows * sizeof(double));
    }

    // Load data into tempValues
    tempValues = Z.getValues();

    // Loop throuch each column (DataFrame is transposed)
    for (int i = 0; i < numCols; ++i) {
        // Find the maximum value in the column
        double maxVal = 0;
        // Loop through each row
        for (int j = 0; j < numRows; ++j) {
            // Keep the maximum value
            maxVal = std::max(maxVal, tempValues[i][j]);
        }
        // Apply exp(val - maxVal) to each element in the column
        for (int j = 0; j < numRows; ++j) {
            tempValues[i][j] = exp(tempValues[i][j] - maxVal);
        }
        // Calculate the sum of the column
        double sumVal = 0;
        for (int j = 0; j < numRows; ++j) {
            sumVal += tempValues[i][j];
        }
        // Divide each element by the sum
        for (int j = 0; j < numRows; ++j) {
            tempValues[i][j] /= sumVal + 0.0000001;
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
