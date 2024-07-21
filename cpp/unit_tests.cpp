#include <gtest/gtest.h>
#include "utils.h"

TEST(DataFrameTest, TransposeTest) {
    int numRows = 2;
    int numCols = 3;
    DataFrame df(numRows, numCols);

    // Allocate and set initial values
    double** initValues = (double**) malloc(numCols * sizeof(double*));
    for (int i = 0; i < numCols; ++i) {
        initValues[i] = (double*) malloc(numRows * sizeof(double));
        for (int j = 0; j < numRows; ++j) {
            initValues[i][j] = i * numRows + j + 1; // Fill with sequential data
        }
    }
    df.setValues(initValues);

    // Transpose the DataFrame
    DataFrame transposed = df.transpose();

    // Check dimensions and values
    ASSERT_EQ(transposed.getNumRows(), numCols);
    ASSERT_EQ(transposed.getNumCols(), numRows);
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            ASSERT_EQ(transposed.getValues()[i][j], initValues[j][i]);
        }
    }

    // Cleanup
    for (int i = 0; i < numCols; ++i) {
        free(initValues[i]);
    }
    free(initValues);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
