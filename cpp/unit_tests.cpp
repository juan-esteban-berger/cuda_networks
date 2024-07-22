#include <gtest/gtest.h>
#include <cmath>

#include "utils.h"
#include "neural_network.h"

////////////////////////////////////////////////////////////////////
// Series Tests
TEST(SeriesTest, SeriesNormalizeTest) {
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

    // Print before normalization
    std::cout << "Before Normalization: " << std::endl;
    series.print(2);
    // Normalize the Series
    series.normalize(min, max);
    // Print after normalization
    std::cout << "After Normalization: " << std::endl;
    series.print(2);

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

TEST(SeriesTest, SeriesDenormalizeTest) {
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

    // Print before denormalization
    std::cout << "Before Denormalization: " << std::endl;
    series.print(2);
    // Denormalize the Series
    series.denormalize(min, max);
    // Print after denormalization
    std::cout << "After Denormalization: " << std::endl;
    series.print(2);

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
TEST(DataFrameTest, DataFrameTransposeTest) {
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

    // Print before transposing
    std::cout << "Before Transposing: " << std::endl;
    df.print(2);
    // Transpose the DataFrame
    DataFrame transposed = df.transpose();
    // Print after transposing
    std::cout << "After Transposing: " << std::endl;
    transposed.print(2);

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

TEST(DataFrameTest, DataFrameNormalizeTest) {
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

    // Print before normalization
    std::cout << "Before Normalization: " << std::endl;
    df.print(2);
    // Normalize the DataFrame
    df.normalize(min, max);
    // Print after normalization
    std::cout << "After Normalization: " << std::endl;
    df.print(2);

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

TEST(DataFrameTest, DataFrameDenormalizeTest) {
    // Number of rows and columns
    int numRows = 2;
    int numCols = 3;
    // Initialize DataFrame
    DataFrame df(numRows, numCols);

    // Allocate memory for normalized values
    double** normalizedValues = (double**) malloc(numCols * sizeof(double*));
    // Initialize the normalized values
    for (int i = 0; i < numCols; ++i) {
        normalizedValues[i] = (double*) malloc(numRows * sizeof(double));
        for (int j = 0; j < numRows; ++j) {
            // Fill with normalized sequential data
            normalizedValues[i][j] = (i * numRows + j) / 5.0;
        }
    }

    // Set normalized values
    df.setValues(normalizedValues);

    // Set min and max values
    double min = 0, max = 5;

    // Print before denormalization
    std::cout << "Before Denormalization: " << std::endl;
    df.print(2);
    // Denormalize the DataFrame
    df.denormalize(min, max);
    // Print after denormalization
    std::cout << "After Denormalization: " << std::endl;
    df.print(2);

    // Allocate memory for 2D array
    double** expectedDenormalized = (double**) malloc(numCols * sizeof(double*));
    // Loop over columns
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each column
        expectedDenormalized[i] = (double*) malloc(numRows * sizeof(double));
        // Loop over rows
        for (int j = 0; j < numRows; ++j) {
            // Fill with sequential data
            expectedDenormalized[i][j] = i * numRows + j;
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
TEST(ActFunctionTest, SigmoidFunctionTest) {
    int numRows = 4;
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
            inputData[i][j] = i * numRows + j + 0.0000001;
            // Expected Data
            expectedData[i][j] = 1.0 / (1.0 + exp(-inputData[i][j]));
        }
    }

    // Set values
    inputDf.setValues(inputData);
    expectedDf.setValues(expectedData);

    // Print before applying Sigmoid function
    std::cout << "Before Sigmoid Function: " << std::endl;
    inputDf.print(2);
    // Apply Sigmoid function
    Sigmoid sigmoid;
    sigmoid.function(inputDf);
    // Print after applying Sigmoid function
    std::cout << "After Sigmoid Function: " << std::endl;
    inputDf.print(2);

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

TEST(ActFunctionTest, SigmoidDerivativeTest) {
    int numRows = 4;
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
            inputData[i][j] = i * numRows + j + 0.0000001;
            // Calculate sigmoid value
            double sigmoidValue = 1.0 / (1.0 + exp(-inputData[i][j]));
            // Expected Data for derivative
            expectedData[i][j] = sigmoidValue * (1 - sigmoidValue);
        }
    }

    // Set values
    inputDf.setValues(inputData);
    expectedDf.setValues(expectedData);

    // Print before applying Sigmoid derivative
    std::cout << "Before Sigmoid Derivative: " << std::endl;
    inputDf.print(2);
    // Apply Sigmoid derivative
    Sigmoid sigmoid;
    sigmoid.derivative(inputDf);
    // Print after applying Sigmoid derivative
    std::cout << "After Sigmoid Derivative: " << std::endl;
    inputDf.print(2);

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

TEST(ActFunctionTest, SoftmaxFunctionTest) {
    int numRows = 10;
    int numCols = 3;
    
    // Initialize DataFrames
    DataFrame inputDf(numRows, numCols), expectedDf(numRows, numCols);

    // Allocate memory for 2D arrays
    double** inputData = (double**) malloc(numCols * sizeof(double*));
    double** expectedData = (double**) malloc(numCols * sizeof(double*));
    
    // Allocate memory for each column
    for (int i = 0; i < numCols; ++i) {
        inputData[i] = (double*) malloc(numRows * sizeof(double));
        expectedData[i] = (double*) malloc(numRows * sizeof(double));
    }

    // Fill in with sequential data
    for (int j = 0; j < numRows; ++j) {
        for (int i = 0; i < numCols; ++i) {
            inputData[i][j] = i * numRows + j - 5;
            expectedData[i][j] = i * numRows + j - 5;
        }
    }

    // Apply Softmax function to expected data
    for (int i = 0; i < numCols; ++i) {
        // Find the maximum value in the column
        double maxVal = 0;
        // Loop through each row
        for (int j = 0; j < numRows; ++j) {
            maxVal = std::max(maxVal, expectedData[i][j]);
        }
        // Apply exp(val - maxVal) to each element in the column
        for (int j = 0; j < numRows; ++j) {
            expectedData[i][j] = exp(expectedData[i][j] - maxVal);
        }
        // Calculate the sum of the column
        double sumVal = 0;
        for (int j = 0; j < numRows; ++j) {
            sumVal += expectedData[i][j];
        }
        // Divide each element by the sum
        for (int j = 0; j < numRows; ++j) {
            expectedData[i][j] /= sumVal + 0.0000001;
        }
    }

    // Set values to DataFrame
    inputDf.setValues(inputData);
    expectedDf.setValues(expectedData);

    // Print before applying Softmax function
    std::cout << "Before Softmax Function: " << std::endl;
    inputDf.print(2);
    // Apply Softmax function
    Softmax softmax;
    softmax.function(inputDf);
    // Print after applying Softmax function
    std::cout << "After Softmax Function: " << std::endl;
    inputDf.print(2);

    // Check values
    double** actualValues = inputDf.getValues();
    double** expectedValues = expectedDf.getValues();

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

////////////////////////////////////////////////////////////////////
// Loss Function Tests
TEST(LossFunctionTest, CategoricalCrossEntropyTest) {
    int numRows = 10;
    int numCols = 3;

    // Initialize DataFrames
    DataFrame Y(numRows, numCols), Y_hat(numRows, numCols);

    // Allocate memory for 2D arrays
    double** YData = (double**) malloc(numCols * sizeof(double*));
    double** Y_hatData = (double**) malloc(numCols * sizeof(double*));

    // Allocate memory for each column
    for (int i = 0; i < numCols; ++i) {
        YData[i] = (double*) malloc(numRows * sizeof(double));
        Y_hatData[i] = (double*) malloc(numRows * sizeof(double));
    }

    // Fill in with sequential data
    for (int j = 0; j < numRows; ++j) {
        for (int i = 0; i < numCols; ++i) {
            // Values between 0 and 1
            YData[i][j] = (i * numRows + j) / 30.0;
            Y_hatData[i][j] = YData[i][j] * 0.75;
        }
    }

    // Calculate the loss
    double crossEntropy = 0;
    // Loop over rows
    for (int i = 0; i < numRows; ++i) {
        // Loop over columns
        for (int j = 0; j < numCols; ++j) {
            crossEntropy -= YData[j][i] * log(Y_hatData[j][i] + 1e-8);
        }
    }

    // Set values to DataFrames
    Y.setValues(YData);
    Y_hat.setValues(Y_hatData);

    // Print DataFrames
    std::cout << "Y: " << std::endl;
    Y.print(2);

    std::cout << "Y_hat: " << std::endl;
    Y_hat.print(2);

    // Print loss
    std::cout << "Loss: " << crossEntropy << std::endl;

    // Calculate the loss
    CatCrossEntropy lossFunc;
    double testCrossEntropy = lossFunc.function(Y, Y_hat);

    // Print test loss
    std::cout << "Test Loss: " << testCrossEntropy << std::endl;

    // Check values
    ASSERT_NEAR(crossEntropy, testCrossEntropy, 1e-5);

    // Free memory
    for (int i = 0; i < numCols; ++i) {
        free(YData[i]);
        free(Y_hatData[i]);
    }
    free(YData);
    free(Y_hatData);
}

////////////////////////////////////////////////////////////////////
// Main Function
int main(int argc, char **argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    // Run all tests
    return RUN_ALL_TESTS();
}
