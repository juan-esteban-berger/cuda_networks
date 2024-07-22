#include <gtest/gtest.h>
#include <cmath>

#include "utils.h"
#include "neural_network.h"

////////////////////////////////////////////////////////////////////
// Series Tests
TEST(UnitTest, SeriesNormalizeTest) {
    // Initialize Series
    Series series(5);
    // Allocate memory for raw values
    double* rawValues = (double*) malloc(5 * sizeof(double));
    // Fill in values
    for (int i = 0; i < 5; i++) {
        rawValues[i] = i + 1;
    }
    // Set values
    series.setValues(rawValues);

    // Set min and max values
    double min = 0, max = 5;

    // Normalize the Series
    series.normalize(min, max);

    // Expected normalized values
    double* expectedNormalized = (double*) malloc(5 * sizeof(double));
    // Fill with normalized data
    for (int i = 0; i < 5; i++) {
        expectedNormalized[i] = (i + 1) / 5.0;
    }
    
    // Iterate over values
    for (int i = 0; i < series.getLength(); i++) {
        // Check values
        ASSERT_NEAR(series.getValues()[i], expectedNormalized[i], 1e-5);
    }

    // Deallocate memory
    free(rawValues);
    free(expectedNormalized);
}

TEST(UnitTest, SeriesDenormalizeTest) {
    // Initialize Series
    Series series(5);
    // Allocate memory for normalized values
    double* normalizedValues = (double*) malloc(5 * sizeof(double));
    // Fill in normalized values
    for (int i = 0; i < 5; i++) {
        normalizedValues[i] = (i + 1) / 5.0;
    }
    // Set normalized values
    series.setValues(normalizedValues);

    // Set min and max values for denormalization
    double min = 0, max = 5;

    // Denormalize the Series
    series.denormalize(min, max);

    // Expected denormalized values
    double* expectedDenormalized = (double*) malloc(5 * sizeof(double));
    // Fill with original data
    for (int i = 0; i < 5; i++) {
        expectedDenormalized[i] = i + 1;
    }

    // Iterate over values to check correctness
    for (int i = 0; i < series.getLength(); i++) {
        // Check values
        ASSERT_NEAR(series.getValues()[i], expectedDenormalized[i], 1e-5);
    }

    // Deallocate memory
    free(normalizedValues);
    free(expectedDenormalized);
}

////////////////////////////////////////////////////////////////////
// DataFrame Tests
TEST(UnitTest, DataFrameTransposeTest) {
    // Number of rows and columns
    int numRows = 2;
    int numCols = 3;

    // Initialize DataFrame
    DataFrame df(numRows, numCols);

    // Allocate memory for 2D array
    double** initValues = (double**) malloc(numCols * sizeof(double*));
    // Iterate over columns
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each column
        initValues[i] = (double*) malloc(numRows * sizeof(double));
        // Iterate over rows
        for (int j = 0; j < numRows; ++j) {
            // Fill with sequential data
            initValues[i][j] = i * numRows + j + 1;
        }
    }
    // Set values
    df.setValues(initValues);

    // Transpose the DataFrame
    DataFrame transposed = df.transpose();

    // Check dimensions
    ASSERT_EQ(transposed.getNumRows(), numCols);
    ASSERT_EQ(transposed.getNumCols(), numRows);

    // Iterate over rows
    for (int i = 0; i < numRows; ++i) {
        // Iterate over columns
        for (int j = 0; j < numCols; ++j) {
            // Check values
            ASSERT_EQ(transposed.getValues()[i][j], initValues[j][i]);
        }
    }

    // Deallocate memory for each row
    for (int i = 0; i < numCols; ++i) {
        free(initValues[i]);
    }
    // Deallocate memory for 2D array
    free(initValues);
}

TEST(UnitTest, DataFrameNormalizeTest) {
    // Number of rows and columns
    int numRows = 2;
    int numCols = 3;

    // Initialize DataFrame
    DataFrame df(numRows, numCols);

    // Allocate memory for 2D array
    double** initValues = (double**) malloc(numCols * sizeof(double*));
    // Iterate over columns
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each column
        initValues[i] = (double*) malloc(numRows * sizeof(double));
        // Iterate over rows
        for (int j = 0; j < numRows; ++j) {
            // Fill with sequential data
            initValues[i][j] = i * numRows + j;
        }
    }
    // Set values
    df.setValues(initValues);

    // Set min and max values for normalization
    double min = 0, max = 5;

    // Normalize the DataFrame
    df.normalize(min, max);

    // Allocate memory for expected normalized values
    double** expectedNormalized = (double**) malloc(numCols * sizeof(double*));
    // Iterate over columns
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each column
        expectedNormalized[i] = (double*) malloc(numRows * sizeof(double));
    }
    // Iterate over columns
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each column
        expectedNormalized[i] = (double*) malloc(numRows * sizeof(double));
        // Iterate over rows
        for (int j = 0; j < numRows; ++j) {
            // Fill with normalized data
            expectedNormalized[i][j] = (i * numRows + j) / 5.0;
        }
    }

    // Iterate over rows and columns to check correctness
    for (int i = 0; i < numCols; ++i) {
        // Iterate over columns
        for (int j = 0; j < numRows; ++j) {
            // Check values
            ASSERT_NEAR(df.getValues()[i][j], expectedNormalized[i][j], 1e-5);
        }
    }

    // Deallocate memory
    for (int i = 0; i < numCols; ++i) {
        free(initValues[i]);
        free(expectedNormalized[i]);
    }
    // Deallocate memory
    free(initValues);
    // Deallocate memory
    free(expectedNormalized);
}

TEST(UnitTest, DataFrameDenormalizeTest) {
    // Number of rows and columns
    int numRows = 2;
    int numCols = 3;
    // Initialize DataFrame
    DataFrame df(numRows, numCols);

    // Allocate memory for normalized values
    double** normalizedValues = (double**) malloc(numCols * sizeof(double*));
    // Iterate over columns
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each column
        normalizedValues[i] = (double*) malloc(numRows * sizeof(double));
        // Iterate over rows
        for (int j = 0; j < numRows; ++j) {
            // Fill with normalized data
            normalizedValues[i][j] = j / 5.0;
        }
    }
    // Set normalized values
    df.setValues(normalizedValues);

    // Set min and max values for denormalization
    double min = 0, max = 5;

    // Denormalize the DataFrame
    df.denormalize(min, max);

    // Allocate memory for expected denormalized values
    double** expectedDenormalized = (double**) malloc(numCols * sizeof(double*));
    // Iterate over columns
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each column
        expectedDenormalized[i] = (double*) malloc(numRows * sizeof(double));
        // Iterate over rows
        for (int j = 0; j < numRows; ++j) {
            // Fill with original data
            expectedDenormalized[i][j] = (j / 5.0) * 5;
        }
    }

    // Iterate over columns
    for (int i = 0; i < numCols; ++i) {
        // Iterate over rows
        for (int j = 0; j < numRows; ++j) {
            // Check values
            ASSERT_NEAR(df.getValues()[i][j], expectedDenormalized[i][j], 1e-5);
        }
    }

    // Deallocate memory
    for (int i = 0; i < numCols; ++i) {
        free(normalizedValues[i]);
        free(expectedDenormalized[i]);
    }
    free(normalizedValues);
    free(expectedDenormalized);
}

////////////////////////////////////////////////////////////////////
// Activation Function Tests
TEST(UnitTest, SigmoidFunctionTest) {
    int numRows = 2;
    int numCols = 3;
    
    // Initialize DataFrames
    DataFrame inputDf(numRows, numCols), expectedDf(numRows, numCols);

    // Initialize 2D arrays
    double** inputData = (double**) malloc(numCols * sizeof(double*));
    double** expectedData = (double**) malloc(numCols * sizeof(double*));
    // Allocate memory for each column
    for (int i = 0; i < numCols; ++i) {
        inputData[i] = (double*) malloc(numRows * sizeof(double));
        expectedData[i] = (double*) malloc(numRows * sizeof(double));
    }
    // Loop over rows
    for (int j = 0; j < numRows; ++j) {
        // Loop over columns
        for (int i = 0; i < numCols; ++i) {
            // Sequential Data
            inputData[i][j] = i * numRows + j - 5;
            // Expected Data
            expectedData[i][j] = 1.0 / (1.0 + exp(-inputData[i][j]));
        }
    }

    // Set values
    inputDf.setValues(inputData);
    expectedDf.setValues(expectedData);

    // Apply Sigmoid function
    Sigmoid sigmoid;
    sigmoid.function(inputDf);

    // Check values
    double** actualValues = inputDf.getValues();
    double** expectedValues = expectedDf.getValues();

    // Check values
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numRows; ++j) {
            ASSERT_NEAR(actualValues[i][j], expectedValues[i][j], 1e-5);
        }
    }

    // Free memory
    for (int i = 0; i < numCols; ++i) {
        free(inputData[i]);
        free(expectedData[i]);
    }
    free(inputData);
    free(expectedData);
}

TEST(UnitTest, SigmoidDerivativeTest) {
    int numRows = 2;
    int numCols = 3;
    
    // Initialize DataFrames
    DataFrame inputDf(numRows, numCols), expectedDf(numRows, numCols);

    // Initialize 2D arrays
    double** inputData = (double**) malloc(numCols * sizeof(double*));
    double** expectedData = (double**) malloc(numCols * sizeof(double*));
    // Allocate memory for each column
    for (int i = 0; i < numCols; ++i) {
        inputData[i] = (double*) malloc(numRows * sizeof(double));
        expectedData[i] = (double*) malloc(numRows * sizeof(double));
    }
    // Loop over rows
    for (int j = 0; j < numRows; ++j) {
        // Loop over columns
        for (int i = 0; i < numCols; ++i) {
            // Sequential Data
            inputData[i][j] = i * numRows + j - 5;
            // Calculate sigmoid value
            double sigmoidValue = 1.0 / (1.0 + exp(-inputData[i][j]));
            // Expected Data for derivative
            expectedData[i][j] = sigmoidValue * (1 - sigmoidValue);
        }
    }

    // Set values
    inputDf.setValues(inputData);
    expectedDf.setValues(expectedData);

    // Apply Sigmoid derivative
    Sigmoid sigmoid;
    sigmoid.derivative(inputDf);

    // Check values
    double** actualValues = inputDf.getValues();
    double** expectedValues = expectedDf.getValues();

    // Check values
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numRows; ++j) {
            ASSERT_NEAR(actualValues[i][j], expectedValues[i][j], 1e-5);
        }
    }

    // Free memory
    for (int i = 0; i < numCols; ++i) {
        free(inputData[i]);
        free(expectedData[i]);
    }
    free(inputData);
    free(expectedData);
}

// TEST(UnitTest, SoftmaxFunctionTest) {
//     int numRows = 2;
//     int numCols = 3;
//     
//     // Initialize DataFrames
//     DataFrame inputDf(numRows, numCols), expectedDf(numRows, numCols);
// 
//     // Initialize 2D arrays
//     double** inputData = (double**) malloc(numCols * sizeof(double*));
//     double** expectedData = (double**) malloc(numCols * sizeof(double*));
//     
//     // Allocate memory and assign values for input and expected output
//     for (int i = 0; i < numCols; ++i) {
//         inputData[i] = (double*) malloc(numRows * sizeof(double));
//         expectedData[i] = (double*) malloc(numRows * sizeof(double));
// 
//         double maxVal = -std::numeric_limits<double>::infinity();
//         double sumExp = 0;
// 
//         for (int j = 0; j < numRows; ++j) {
//             // Sequential data for testing
//             inputData[i][j] = i * numRows + j - 5;
// 
//             // Calculate max for stability in softmax
//             if (inputData[i][j] > maxVal) {
//                 maxVal = inputData[i][j];
//             }
//         }
// 
//         for (int j = 0; j < numRows; ++j) {
//             // Calculate exp(values - maxVal)
//             expectedData[i][j] = exp(inputData[i][j] - maxVal);
//             sumExp += expectedData[i][j];
//         }
// 
//         for (int j = 0; j < numRows; ++j) {
//             // Normalize to get probabilities
//             expectedData[i][j] /= sumExp + 0.0000001; // Add a small constant for numerical stability
//         }
//     }
// 
//     // Set values to DataFrame
//     inputDf.setValues(inputData);
//     expectedDf.setValues(expectedData);
// 
//     // Apply Softmax function
//     Softmax softmax;
//     softmax.function(inputDf);
// 
//     // Check values
//     double** actualValues = inputDf.getValues();
//     double** expectedValues = expectedDf.getValues();
// 
//     for (int i = 0; i < numCols; ++i) {
//         for (int j = 0; j < numRows; ++j) {
//             ASSERT_NEAR(actualValues[i][j], expectedValues[i][j], 1e-5);
//         }
//     }
// 
//     // Free memory
//     for (int i = 0; i < numCols; ++i) {
//         free(inputData[i]);
//         free(expectedData[i]);
//     }
//     free(inputData);
//     free(expectedData);
// }

////////////////////////////////////////////////////////////////////
// Loss Function Tests

////////////////////////////////////////////////////////////////////
// Main Function
int main(int argc, char **argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    // Run all tests
    return RUN_ALL_TESTS();
}
