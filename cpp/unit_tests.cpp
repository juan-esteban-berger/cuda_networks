#include <gtest/gtest.h>
#include "utils.h"

////////////////////////////////////////////////////////////////////
// Series Tests

////////////////////////////////////////////////////////////////////
// DataFrame Tests
TEST(DataFrameTest, TransposeTest) {
    // Number of rows and columns
    int numRows = 2;
    int numCols = 3;

    // Initialize DataFrame
    DataFrame df(numRows, numCols);

    // Allocate memory for 2D array
    double** initValues = (double**) malloc(numCols * sizeof(double*));
    // Iterate over columns
    for (int i = 0; i < numCols; ++i) {
        // Allocate memory for each row
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

////////////////////////////////////////////////////////////////////
// Main Function
int main(int argc, char **argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    // Run all tests
    return RUN_ALL_TESTS();
}
