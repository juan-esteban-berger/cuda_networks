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
    for (int i = 0; i < numCols; ++i) {
        tempValues[i] = (double*) malloc(numRows * sizeof(double));
    }

    // Load data into tempValues from the original values in Z
    double** originalValues = Z.getValues();
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numRows; ++j) {
            tempValues[i][j] = originalValues[i][j];
        }
    }

    // Process each column for Softmax
    for (int i = 0; i < numCols; ++i) {
        // Initialize maxVal to the first element of the column
        double maxVal = tempValues[i][0];
        // Find the max value in the column
        for (int j = 1; j < numRows; ++j) {
            if (tempValues[i][j] > maxVal) {
                maxVal = tempValues[i][j];
            }
        }

        // Sum all the exponentiated values
        double sumVal = 0;
        // Apply exp(val - maxVal) to each element in the column
        for (int j = 0; j < numRows; ++j) {
            tempValues[i][j] = exp(tempValues[i][j] - maxVal);
            sumVal += tempValues[i][j];
        }

        // Normalize by dividing by the sum
        for (int j = 0; j < numRows; ++j) {
            tempValues[i][j] /= sumVal;
        }
    }

    // Set the processed values back to Z
    Z.setValues(tempValues);

    // Free temporary array
    for (int i = 0; i < numCols; ++i) {
        free(tempValues[i]);
    }
    free(tempValues);
}

////////////////////////////////////////////////////////////////////
// Loss Function Classes
double CatCrossEntropy::function(DataFrame& Y, DataFrame& Y_hat) {
    int numRows = Y.getNumRows();
    int numCols = Y.getNumCols();

    // Allocate temporary copies for computation
    double** tempY = (double**) malloc(numRows * sizeof(double*));
    double** tempYHat = (double**) malloc(numRows * sizeof(double*));

    // Loop through the rows
    for (int i = 0; i < numRows; ++i) {
        tempY[i] = (double*) malloc(numCols * sizeof(double));
        tempYHat[i] = (double*) malloc(numCols * sizeof(double));
        for (int j = 0; j < numCols; ++j) {
            tempY[i][j] = Y.getValues()[j][i];
            tempYHat[i][j] = Y_hat.getValues()[j][i];
        }
    }

    // Calculate the loss
    double crossEntropy = 0.0;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            crossEntropy -= tempY[i][j] * log(tempYHat[i][j] + 1e-8);
        }
    }

    // Free the temporary arrays
    for (int i = 0; i < numRows; ++i) {
        free(tempY[i]);
        free(tempYHat[i]);
    }
    free(tempY);
    free(tempYHat);

    return crossEntropy;
}
