#include <iostream>
#include <cmath>
#include <gtest/gtest.h>

#include "linear_algebra.h"
#include "neural_network.h"

//////////////////////////////////////////////////////////////////
// Transpose Tests
TEST(TransposeTest, MatrixTransposeTest) {
    // Initialize Matrix
    Matrix matrix(2, 3);
    
    // Fill in values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            matrix.setValue(i, j, i * 3 + j);
        }
    }

    std::cout << "Original Matrix:" << std::endl;
    preview_matrix(&matrix, 2);

    // Transpose the Matrix
    Matrix* transposed = transpose_matrix(&matrix);

    std::cout << "Transposed Matrix:" << std::endl;
    preview_matrix(transposed, 2);

    // Check dimensions
    ASSERT_EQ(transposed->rows, matrix.cols);
    ASSERT_EQ(transposed->cols, matrix.rows);

    // Check values
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            ASSERT_EQ(matrix.getValues(i, j), transposed->getValues(j, i));
        }
    }

    // Clean up
    delete transposed;
}

//////////////////////////////////////////////////////////////////
// Normalization Tests
TEST(NormalizeTest, VectorNormalizeTest) {
    // Initialize Vector
    Vector vector(5);
    // Allocate memory for raw values
    double* rawValues = new double[5];
    // Fill in values
    for (int i = 0; i < 5; i++) {
        rawValues[i] = i;
        vector.setValue(i, rawValues[i]);
    }

    // Set min and max values
    double min = 0, max = 5;

    // Print before normalization
    std::cout << "Before Normalization: " << std::endl;
    preview_vector(&vector, 2);

    // Normalize the Vector
    normalize_vector(&vector, min, max);

    // Print after normalization
    std::cout << "After Normalization: " << std::endl;
    preview_vector(&vector, 2);

    // Expected normalized values
    double* expectedNormalized = new double[5];
    // Fill with normalized data
    for (int i = 0; i < 5; i++) {
        expectedNormalized[i] = (i) / 5.0;
    }
    
    // Iterate over values
    for (int i = 0; i < 5; i++) {
        // Check values
        ASSERT_NEAR(vector.getValues(i), expectedNormalized[i], 1e-5);
    }

    // Deallocate memory
    delete[] rawValues;
    delete[] expectedNormalized;
}

TEST(NormalizeTest, VectorDenormalizeTest) {
    // Initialize Vector
    Vector vector(5);
    // Allocate memory for normalized values
    double* normalizedValues = new double[5];
    // Fill in normalized values
    for (int i = 0; i < 5; i++) {
        normalizedValues[i] = (i) / 5.0;
        vector.setValue(i, normalizedValues[i]);
    }

    // Set min and max values for denormalization
    double min = 0, max = 5;

    // Print before denormalization
    std::cout << "Before Denormalization: " << std::endl;
    preview_vector(&vector, 2);

    // Denormalize the Vector
    denormalize_vector(&vector, min, max);

    // Print after denormalization
    std::cout << "After Denormalization: " << std::endl;
    preview_vector(&vector, 2);

    // Expected denormalized values
    double* expectedDenormalized = new double[5];
    // Fill with original data
    for (int i = 0; i < 5; i++) {
        expectedDenormalized[i] = i;
    }

    // Iterate over values to check correctness
    for (int i = 0; i < 5; i++) {
        // Check values
        ASSERT_NEAR(vector.getValues(i), expectedDenormalized[i], 1e-5);
    }

    // Deallocate memory
    delete[] normalizedValues;
    delete[] expectedDenormalized;
}

TEST(NormalizeTest, MatrixNormalizeTest) {
    // Initialize Matrix
    Matrix matrix(2, 3);
    // Allocate memory for raw values
    double** rawValues = new double*[2];
    for (int i = 0; i < 2; i++) {
        rawValues[i] = new double[3];
    }
    // Fill in values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            rawValues[i][j] = i * 3 + j;
            matrix.setValue(i, j, rawValues[i][j]);
        }
    }

    // Set min and max values
    double min = 0, max = 5;

    // Print before normalization
    std::cout << "Before Normalization: " << std::endl;
    preview_matrix(&matrix, 2);

    // Normalize the Matrix
    normalize_matrix(&matrix, min, max);

    // Print after normalization
    std::cout << "After Normalization: " << std::endl;
    preview_matrix(&matrix, 2);

    // Expected normalized values
    double** expectedNormalized = new double*[2];
    for (int i = 0; i < 2; i++) {
        expectedNormalized[i] = new double[3];
        for (int j = 0; j < 3; j++) {
            expectedNormalized[i][j] = (i * 3 + j) / 5.0;
        }
    }
    
    // Iterate over values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            // Check values
            ASSERT_NEAR(matrix.getValues(i, j), expectedNormalized[i][j], 1e-5);
        }
    }

    // Deallocate memory
    for (int i = 0; i < 2; i++) {
        delete[] rawValues[i];
        delete[] expectedNormalized[i];
    }
    delete[] rawValues;
    delete[] expectedNormalized;
}

TEST(NormalizeTest, MatrixDenormalizeTest) {
    // Initialize Matrix
    Matrix matrix(2, 3);
    // Allocate memory for normalized values
    double** normalizedValues = new double*[2];
    for (int i = 0; i < 2; i++) {
        normalizedValues[i] = new double[3];
    }
    // Fill in normalized values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            normalizedValues[i][j] = (i * 3 + j) / 5.0;
            matrix.setValue(i, j, normalizedValues[i][j]);
        }
    }

    // Set min and max values for denormalization
    double min = 0, max = 5;

    // Print before denormalization
    std::cout << "Before Denormalization: " << std::endl;
    preview_matrix(&matrix, 2);

    // Denormalize the Matrix
    denormalize_matrix(&matrix, min, max);

    // Print after denormalization
    std::cout << "After Denormalization: " << std::endl;
    preview_matrix(&matrix, 2);

    // Expected denormalized values
    double** expectedDenormalized = new double*[2];
    for (int i = 0; i < 2; i++) {
        expectedDenormalized[i] = new double[3];
        for (int j = 0; j < 3; j++) {
            expectedDenormalized[i][j] = i * 3 + j;
        }
    }

    // Iterate over values to check correctness
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            // Check values
            ASSERT_NEAR(matrix.getValues(i, j), expectedDenormalized[i][j], 1e-5);
        }
    }

    // Deallocate memory
    for (int i = 0; i < 2; i++) {
        delete[] normalizedValues[i];
        delete[] expectedDenormalized[i];
    }
    delete[] normalizedValues;
    delete[] expectedDenormalized;
}

//////////////////////////////////////////////////////////////////
// Activation Function Tests
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

TEST(ActFuncTest, FunctionTest) {
    // Initialize Matrix with sequential data
    Matrix matrix(2, 2);
    double** rawValues = new double*[2];
    for (int i = 0; i < 2; i++) {
        rawValues[i] = new double[2];
        for (int j = 0; j < 2; j++) {
            rawValues[i][j] = i * 2 + j;
            matrix.setValue(i, j, rawValues[i][j]);
        }
    }

    // Expected results using sigmoid
    Matrix expected(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            expected.setValue(i, j, sigmoid(rawValues[i][j]));
        }
    }

    // Preview before applying Sigmoid function
    std::cout << "Before Sigmoid Function: " << std::endl;
    preview_matrix(&matrix, 2);
    // Apply Sigmoid function
    Sigmoid sig;
    sig.function(matrix);
    // Preview after applying Sigmoid function
    std::cout << "After Sigmoid Function: " << std::endl;
    preview_matrix(&matrix, 2);

    // Check each element
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            ASSERT_NEAR(matrix.getValues(i, j), expected.getValues(i, j), 1e-5);
        }
    }

    // Deallocate memory
    for (int i = 0; i < 2; i++) {
        delete[] rawValues[i];
    }
    delete[] rawValues;
}

double sigmoid_derivative(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

TEST(ActFuncTest, DerivativeTest) {
    // Initialize Matrix with sequential data
    Matrix matrix(2, 2);
    double** rawValues = new double*[2];
    for (int i = 0; i < 2; i++) {
        rawValues[i] = new double[2];
        for (int j = 0; j < 2; j++) {
            rawValues[i][j] = i * 2 + j;
            matrix.setValue(i, j, rawValues[i][j]);
        }
    }

    // Expected results using sigmoid_derivative
    Matrix expected(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            expected.setValue(i, j, sigmoid_derivative(rawValues[i][j]));
        }
    }

    // Preview before applying Sigmoid derivative
    std::cout << "Before Sigmoid Derivative: " << std::endl;
    preview_matrix(&matrix, 2);
    // Apply Sigmoid derivative
    Sigmoid sig;
    sig.derivative(matrix);
    // Preview after applying Sigmoid derivative
    std::cout << "After Sigmoid Derivative: " << std::endl;
    preview_matrix(&matrix, 2);

    // Check each element
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            ASSERT_NEAR(matrix.getValues(i, j), expected.getValues(i, j), 1e-5);
        }
    }

    // Deallocate memory
    for (int i = 0; i < 2; i++) {
        delete[] rawValues[i];
    }
    delete[] rawValues;
}

//////////////////////////////////////////////////////////////////
// Loss Function Tests
double cat_cross_entropy(double** Y,
                         double** Y_hat,
                         int rows, int cols) {
    double loss = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            loss -= Y[i][j] * log(Y_hat[i][j] + 1e-8);
        }
    }
    return loss;
}

TEST(LossFuncTest, CatCrossEntropyTest) {
    int rows = 3, cols = 4;

    // Initialize matrices
    Matrix Y(rows, cols), Y_hat(rows, cols);
    
    // Allocate 2D arrays for raw values
    double** rawValuesY = new double*[rows];
    double** rawValuesYHat = new double*[rows];
    for (int i = 0; i < rows; i++) {
        rawValuesY[i] = new double[cols];
        rawValuesYHat[i] = new double[cols];
    }

    // Simulate Data
    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = 0; j < cols; j++) {
            // One-hot encoding for Y
            if (j % rows == i) {
                rawValuesY[i][j] = 1.0;
            } else {
                rawValuesY[i][j] = 0.0;
            }
            
            // Arbitrary positive values for Y_hat
            rawValuesYHat[i][j] = 1.0 + j;
            sum += rawValuesYHat[i][j];
        }
        // Normalize Y_hat
        for (int j = 0; j < cols; j++) {
            rawValuesYHat[i][j] /= sum;
            Y.setValue(i, j, rawValuesY[i][j]);
            Y_hat.setValue(i, j, rawValuesYHat[i][j]);
        }
    }

    // Preview matrices
    std::cout << "Matrix Y (True Labels):" << std::endl;
    preview_matrix(&Y, 3);
    std::cout << "Matrix Y_hat (Predictions):" << std::endl;
    preview_matrix(&Y_hat, 3);

    // Calculate loss
    CatCrossEntropy ce;
    double loss = ce.function(Y, Y_hat);
    double expected_loss = cat_cross_entropy(rawValuesY,
                                             rawValuesYHat,
                                             rows, cols);

    // Print the calculated and expected losses
    std::cout << "Calculated Loss from Class: " << loss << std::endl;
    std::cout << "Expected Loss from Helper Function: " << expected_loss << std::endl;
    
    // Check correctness
    ASSERT_NEAR(loss, expected_loss, 1e-5);
    
    // Deallocate memory
    for (int i = 0; i < rows; i++) {
        delete[] rawValuesY[i];
        delete[] rawValuesYHat[i];
    }
    delete[] rawValuesY;
    delete[] rawValuesYHat;
}
