#include <iostream>
#include <cmath>
#include <memory>
#include <gtest/gtest.h>

#include "linear_algebra.h"
#include "neural_network.h"

//////////////////////////////////////////////////////////////////
// Matrix and Vector Operations Tests
// Test for element-wise multiplication (*)
TEST(OperatorTest, ElementWiseMultiplicationTest) {
    // Initialize matrices
    Matrix m1(2, 2);
    Matrix m2(2, 2);
    
    // Fill matrices with values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            m1.setValue(i, j, i * 2 + j + 1);  // m1 will be [1 2; 3 4]
            m2.setValue(i, j, (i * 2 + j + 1) * 2);  // m2 will be [2 4; 6 8]
        }
    }

    // Preview matrices before multiplication
    std::cout << "Matrix m1 before multiplication:" << std::endl;
    preview_matrix(&m1, 2);
    std::cout << "Matrix m2 before multiplication:" << std::endl;
    preview_matrix(&m2, 2);

    // Perform element-wise multiplication
    Matrix result = m1 * m2;

    // Preview result after multiplication
    std::cout << "Result of m1 * m2:" << std::endl;
    preview_matrix(&result, 2);

    // Check results
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            ASSERT_EQ(result.getValues(i, j),
                      m1.getValues(i, j) * m2.getValues(i, j));
        }
    }
}

// Test for matrix multiplication
TEST(OperatorTest, MatrixMultiplicationTest) {
    // Initialize matrices
    Matrix m1(2, 3);
    Matrix m2(3, 2);
    
    // Fill matrices with values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            m1.setValue(i, j, i * 3 + j + 1);  // m1 will be [1 2 3; 4 5 6]
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            m2.setValue(i, j, i * 2 + j + 1);  // m2 will be [1 2; 3 4; 5 6]
        }
    }

    // Preview matrices before multiplication
    std::cout << "Matrix m1 before multiplication:" << std::endl;
    preview_matrix(&m1, 2);
    std::cout << "Matrix m2 before multiplication:" << std::endl;
    preview_matrix(&m2, 2);

    // Perform matrix multiplication
    Matrix result = matmul(m1, m2);

    // Preview result after multiplication
    std::cout << "Result of matmul(m1, m2):" << std::endl;
    preview_matrix(&result, 2);

    // Expected result: [22 28; 49 64]
    ASSERT_EQ(result.getValues(0, 0), 22);
    ASSERT_EQ(result.getValues(0, 1), 28);
    ASSERT_EQ(result.getValues(1, 0), 49);
    ASSERT_EQ(result.getValues(1, 1), 64);
}

// Test for matrix-vector addition (+)
TEST(OperatorTest, MatrixVectorAdditionTest) {
    // Initialize matrix and vector
    Matrix m(2, 3);
    Vector v(2);
    
    // Fill matrix and vector with values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            m.setValue(i, j, i * 3 + j + 1);  // m will be [1 2 3; 4 5 6]
        }
        v.setValue(i, i + 1);  // v will be [1; 2]
    }

    // Preview matrix and vector before addition
    std::cout << "Matrix m before addition:" << std::endl;
    preview_matrix(&m, 2);
    std::cout << "Vector v before addition:" << std::endl;
    preview_vector(&v, 2);

    // Perform matrix-vector addition
    Matrix result = m + v;

    // Preview result after addition
    std::cout << "Result of m + v:" << std::endl;
    preview_matrix(&result, 2);

    // Check results
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            ASSERT_EQ(result.getValues(i, j),
                      m.getValues(i, j) + v.getValues(i));
        }
    }
}

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
// Matrix Argmax Test
// Test for Argmax of each column
TEST(ArgmaxTest, ColumnArgmaxTest) {
    // Initialize matrix with specific values
    Matrix m(3, 2); // 3 rows, 2 columns
    m.setValue(0, 0, 1.0);
    m.setValue(1, 0, 3.0);
    m.setValue(2, 0, 2.0);
    m.setValue(0, 1, 4.0);
    m.setValue(1, 1, 1.0);
    m.setValue(2, 1, 3.0);

    // Preview matrix before argmax
    std::cout << "Before Argmax: " << std::endl;
    preview_matrix(&m, 2);

    // Calculate argmax
    Vector result = argmax(m);

    // Preview result after argmax
    std::cout << "Result of Argmax: " << std::endl;
    preview_vector(&result, 2);

    // Check results
    ASSERT_EQ(result.getValues(0), 1);
    ASSERT_EQ(result.getValues(1), 0);
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

TEST(ActFuncTest, SoftmaxFunctionTest) {
    // Initialize Matrix for raw
    Matrix matrix(3, 2);
    // Allocate 2D array for raw values
    double** rawValues = new double*[3];
    // Loop over rows
    for (int i = 0; i < 3; i++) {
        // Allocate memory for each row
        rawValues[i] = new double[2];
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            // Fill with sequential data
            rawValues[i][j] = i * 2 + j;
            matrix.setValue(i, j, rawValues[i][j]);
        }
    }

    // Initialize Matrix for expected
    Matrix expected(3, 2);
    // Loop over columns
    for (int j = 0; j < 2; j++) {
        // Set max value to first element
        double max_val = rawValues[0][j];
        // Loop over rows
        for (int i = 1; i < 3; i++) {
            // Find max value
            double temp_val = rawValues[i][j];
            if (temp_val > max_val) {
                max_val = temp_val;
            }
        }
        
        // Compute exp(Z - max)
        for (int i = 0; i < 3; i++) {
            double exp_val = std::exp(rawValues[i][j] - max_val);
            expected.setValue(i, j, exp_val);
        }

        // Compute sum(exp(Z - max))
        double sum = 0.0;
        for (int i = 0; i < 3; i++) {
            sum += expected.getValues(i, j);
        }

        // Divide by sum
        for (int i = 0; i < 3; i++) {
            expected.setValue(i, j,
                              expected.getValues(i, j) / (sum + 1e-8));
        }
    }

    // Preview before applying Softmax function
    std::cout << "Before Softmax Function: " << std::endl;
    preview_matrix(&matrix, 4);
    
    // Apply Softmax function
    Softmax softmax;
    softmax.function(matrix);
    
    // Preview after applying Softmax function
    std::cout << "After Softmax Function: " << std::endl;
    preview_matrix(&matrix, 4);

    // Check each element
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            ASSERT_NEAR(matrix.getValues(i, j),
                        expected.getValues(i, j), 1e-5);
        }
    }

    // Deallocate memory
    for (int i = 0; i < 3; i++) {
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

//////////////////////////////////////////////////////////////////
// Forward Propagation Test
TEST(NeuralNetworkTest, ForwardPropagationTest) {
//////////////////////////////////////////////////////////////////
    // Initialize input matrix
    Matrix X(3, 2);  // 3 features, 2 examples
    // Initialize 2D array for inputs
    double X_values[3][2] = {{1, 4}, {2, 5}, {3, 6}};
    // Loop over rows
    for (int i = 0; i < 3; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            X.setValue(i, j, X_values[i][j]);
        }
    }

    // Preview input matrix
    std::cout << "Input Matrix X:" << std::endl;
    preview_matrix(&X, 2);

//////////////////////////////////////////////////////////////////
    // Create a simple neural network
    NeuralNetwork nn;
    Layer* layer1 = new Layer(3, 2, "Sigmoid");
    Layer* layer2 = new Layer(2, 2, "Softmax");

//////////////////////////////////////////////////////////////////
    // Initialize 2D arrays for weights
    double W1_values[2][3] = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    // Initialize 1D arrays for biases
    double b1_values[2] = {0.1, 0.2};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 3; j++) {
            layer1->W->setValue(i, j, W1_values[i][j]);
        }
        layer1->b->setValue(i, b1_values[i]);
    }

    // Preview layer 1 weights
    std::cout << "Layer 1 Weights:" << std::endl;
    preview_matrix(layer1->W, 4);

    // Preview layer 1 biases
    std::cout << "Layer 1 Biases:" << std::endl;
    preview_vector(layer1->b, 4);

//////////////////////////////////////////////////////////////////
    // Initialize 2D arrays for weights
    double W2_values[2][2] = {{0.7, 0.8}, {0.9, 1.0}};
    // Initialize 1D arrays for biases
    double b2_values[2] = {0.3, 0.4};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            layer2->W->setValue(i, j, W2_values[i][j]);
        }
        layer2->b->setValue(i, b2_values[i]);
    }

    // Preview layer 2 weights
    std::cout << "Layer 2 Weights:" << std::endl;
    preview_matrix(layer2->W, 4);

    // Preview layer 2 biases
    std::cout << "Layer 2 Biases:" << std::endl;
    preview_vector(layer2->b, 4);

//////////////////////////////////////////////////////////////////
    // Calculate Z1
    // Multiply weights with input
    Matrix Z1_temp = matmul(*layer1->W, X);
    // Initialize matrix for Z1
    Matrix Z1(Z1_temp.rows, Z1_temp.cols);
    // Loop over rows
    for (int i = 0; i < Z1.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z1.cols; j++) {
            // Add bias to the result
            Z1.setValue(i, j,
                        Z1_temp.getValues(i, j) + layer1->b->getValues(i));
        }
    }
    // Preview Z1
    std::cout << "Manual Z1:" << std::endl;
    preview_matrix(&Z1, 4);
    
//////////////////////////////////////////////////////////////////
    // Calculate A1
    Matrix A1(Z1.rows, Z1.cols);
    // Loop over rows
    for (int i = 0; i < Z1.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z1.cols; j++) {
            // Set values
            A1.setValue(i, j, Z1.getValues(i, j));
        }
    }
    // Initialize Sigmoid object
    Sigmoid sigmoid;
    // Apply Sigmoid function
    sigmoid.function(A1);

    // Preview Activated Z1
    std::cout << "Activated Z1:" << std::endl;
    preview_matrix(&A1, 4);

//////////////////////////////////////////////////////////////////
    // Calculate Z2
    // Multiply weights with A1
    Matrix Z2_temp = matmul(*layer2->W, A1);
    // Initialize matrix for Z2
    Matrix Z2(Z2_temp.rows, Z2_temp.cols);
    // Loop over rows
    for (int i = 0; i < Z2.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z2.cols; j++) {
            // Add bias to the result
            Z2.setValue(i, j,
                        Z2_temp.getValues(i, j) + layer2->b->getValues(i));
        }
    }

    // Preview Z2
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&Z2, 4);

//////////////////////////////////////////////////////////////////
    // Calculate A2
    Matrix A2(Z2.rows, Z2.cols);
    // Loop over rows
    for (int i = 0; i < Z2.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z2.cols; j++) {
            // Set values
            A2.setValue(i, j, Z2.getValues(i, j));
        }
    }

    // Initialize Softmax object
    Softmax softmax;
    // Apply Softmax function
    softmax.function(A2);

    // Preview A2
    std::cout << "Activated Z2 (final output):" << std::endl;
    preview_matrix(&A2, 4);

//////////////////////////////////////////////////////////////////
    // Test Forward Function
    // Add layers to the neural network
    nn.add_layer(layer1);
    nn.add_layer(layer2);

    // Run forward propagation
    nn.forward(X);

//////////////////////////////////////////////////////////////////
    // Check Z and A Matrices
    // Get Z1 from neural network
    Matrix* Z1_nn = nn.layers[0]->Z;
    
    // Preview Z1 from neural network
    std::cout << "Neural Network Z1:" << std::endl;
    preview_matrix(Z1_nn, 4);

    // Preview Z1 from Test
    std::cout << "Manual Z1:" << std::endl;
    preview_matrix(&Z1, 4);

    // Get A1 from neural network
    Matrix* A1_nn = nn.layers[0]->A;

    // Preview A1 from neural network
    std::cout << "Neural Network A1:" << std::endl;
    preview_matrix(A1_nn, 4);

    // Preview A1 from Test
    std::cout << "Manual A1:" << std::endl;
    preview_matrix(&A1, 4);

    // Get Z2 from neural network
    Matrix* Z2_nn = nn.layers[1]->Z;

    // Preview Z2 from neural network
    std::cout << "Neural Network Z2:" << std::endl;
    preview_matrix(Z2_nn, 4);

    // Preview Z2 from Test
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&Z2, 4);

    // Preview A2 from neural network
    Matrix* A2_nn = nn.layers[1]->A;

    // Preview A2 from neural network
    std::cout << "Neural Network A2:" << std::endl;
    preview_matrix(A2_nn, 4);

    // Preview A2 from Test
    std::cout << "Manual A2:" << std::endl;
    preview_matrix(&A2, 4);

//////////////////////////////////////////////////////////////////
    // Check output values
    // Get the output of the last layer
    Matrix* output = nn.getOutput();

    // Preview Neural Network output
    std::cout << "Neural Network output:" << std::endl;
    preview_matrix(output, 4);

    // Preview output from test
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&A2, 4);

    // Check values
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++) {
            ASSERT_NEAR(output->getValues(i, j),
                        A2.getValues(i, j), 1e-5);
        }
    }
}

TEST(NeuralNetworkTest, BackwardPropagationTest) {
//////////////////////////////////////////////////////////////////
    // Forward Propagation Section
//////////////////////////////////////////////////////////////////
    // Initialize input matrix
    Matrix X(3, 2);  // 3 features, 2 examples
    // Initialize 2D array for inputs
    double X_values[3][2] = {{1, 4}, {2, 5}, {3, 6}};
    // Loop over rows
    for (int i = 0; i < 3; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            X.setValue(i, j, X_values[i][j]);
        }
    }

    // Preview input matrix
    std::cout << "Input Matrix X:" << std::endl;
    preview_matrix(&X, 2);

//////////////////////////////////////////////////////////////////
    // Create a simple neural network
    NeuralNetwork nn;
    Layer* layer1 = new Layer(3, 2, "Sigmoid");
    Layer* layer2 = new Layer(2, 2, "Softmax");

//////////////////////////////////////////////////////////////////
    // Initialize 2D arrays for weights
    double W1_values[2][3] = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    // Initialize 1D arrays for biases
    double b1_values[2] = {0.1, 0.2};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 3; j++) {
            layer1->W->setValue(i, j, W1_values[i][j]);
        }
        layer1->b->setValue(i, b1_values[i]);
    }

    // Preview layer 1 weights
    std::cout << "Layer 1 Weights:" << std::endl;
    preview_matrix(layer1->W, 4);

    // Preview layer 1 biases
    std::cout << "Layer 1 Biases:" << std::endl;
    preview_vector(layer1->b, 4);

//////////////////////////////////////////////////////////////////
    // Initialize 2D arrays for weights
    double W2_values[2][2] = {{0.7, 0.8}, {0.9, 1.0}};
    // Initialize 1D arrays for biases
    double b2_values[2] = {0.3, 0.4};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            layer2->W->setValue(i, j, W2_values[i][j]);
        }
        layer2->b->setValue(i, b2_values[i]);
    }

    // Preview layer 2 weights
    std::cout << "Layer 2 Weights:" << std::endl;
    preview_matrix(layer2->W, 4);

    // Preview layer 2 biases
    std::cout << "Layer 2 Biases:" << std::endl;
    preview_vector(layer2->b, 4);

//////////////////////////////////////////////////////////////////
    // Calculate Z1
    // Multiply weights with input
    Matrix Z1_temp = matmul(*layer1->W, X);
    // Initialize matrix for Z1
    Matrix Z1(Z1_temp.rows, Z1_temp.cols);
    // Loop over rows
    for (int i = 0; i < Z1.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z1.cols; j++) {
            // Add bias to the result
            Z1.setValue(i, j,
                        Z1_temp.getValues(i, j) + layer1->b->getValues(i));
        }
    }
    // Preview Z1
    std::cout << "Manual Z1:" << std::endl;
    preview_matrix(&Z1, 4);
    
//////////////////////////////////////////////////////////////////
    // Calculate A1
    Matrix A1(Z1.rows, Z1.cols);
    // Loop over rows
    for (int i = 0; i < Z1.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z1.cols; j++) {
            // Set values
            A1.setValue(i, j, Z1.getValues(i, j));
        }
    }
    // Initialize Sigmoid object
    Sigmoid sigmoid;
    // Apply Sigmoid function
    sigmoid.function(A1);

    // Preview Activated Z1
    std::cout << "Activated Z1:" << std::endl;
    preview_matrix(&A1, 4);

//////////////////////////////////////////////////////////////////
    // Calculate Z2
    // Multiply weights with A1
    Matrix Z2_temp = matmul(*layer2->W, A1);
    // Initialize matrix for Z2
    Matrix Z2(Z2_temp.rows, Z2_temp.cols);
    // Loop over rows
    for (int i = 0; i < Z2.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z2.cols; j++) {
            // Add bias to the result
            Z2.setValue(i, j,
                        Z2_temp.getValues(i, j) + layer2->b->getValues(i));
        }
    }

    // Preview Z2
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&Z2, 4);

//////////////////////////////////////////////////////////////////
    // Calculate A2
    Matrix A2(Z2.rows, Z2.cols);
    // Loop over rows
    for (int i = 0; i < Z2.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z2.cols; j++) {
            // Set values
            A2.setValue(i, j, Z2.getValues(i, j));
        }
    }

    // Initialize Softmax object
    Softmax softmax;
    // Apply Softmax function
    softmax.function(A2);

    // Preview A2
    std::cout << "Activated Z2 (final output):" << std::endl;
    preview_matrix(&A2, 4);

//////////////////////////////////////////////////////////////////
    // Test Forward Function
    // Add layers to the neural network
    nn.add_layer(layer1);
    nn.add_layer(layer2);

    // Run forward propagation
    nn.forward(X);

//////////////////////////////////////////////////////////////////
    // Check Z and A Matrices
    // Get Z1 from neural network
    Matrix* Z1_nn = nn.layers[0]->Z;
    
    // Preview Z1 from neural network
    std::cout << "Neural Network Z1:" << std::endl;
    preview_matrix(Z1_nn, 4);

    // Preview Z1 from Test
    std::cout << "Manual Z1:" << std::endl;
    preview_matrix(&Z1, 4);

    // Get A1 from neural network
    Matrix* A1_nn = nn.layers[0]->A;

    // Preview A1 from neural network
    std::cout << "Neural Network A1:" << std::endl;
    preview_matrix(A1_nn, 4);

    // Preview A1 from Test
    std::cout << "Manual A1:" << std::endl;
    preview_matrix(&A1, 4);

    // Get Z2 from neural network
    Matrix* Z2_nn = nn.layers[1]->Z;

    // Preview Z2 from neural network
    std::cout << "Neural Network Z2:" << std::endl;
    preview_matrix(Z2_nn, 4);

    // Preview Z2 from Test
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&Z2, 4);

    // Preview A2 from neural network
    Matrix* A2_nn = nn.layers[1]->A;

    // Preview A2 from neural network
    std::cout << "Neural Network A2:" << std::endl;
    preview_matrix(A2_nn, 4);

    // Preview A2 from Test
    std::cout << "Manual A2:" << std::endl;
    preview_matrix(&A2, 4);

//////////////////////////////////////////////////////////////////
    // Check output values
    // Get the output of the last layer
    Matrix* output = nn.getOutput();

    // Preview Neural Network output
    std::cout << "Neural Network output:" << std::endl;
    preview_matrix(output, 4);

    // Preview output from test
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&A2, 4);

    // Check values
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++) {
            ASSERT_NEAR(output->getValues(i, j),
                        A2.getValues(i, j), 1e-5);
        }
    }

//////////////////////////////////////////////////////////////////
    // Backward Propagation Section
    // Print divider
    std::cout << "________________________________" << std::endl;
//////////////////////////////////////////////////////////////////
    // Initialize output matrix
    Matrix Y(2, 2);  // 2 outputs, 2 examples
    // Initialize 2D array for inputs
    double Y_values[3][2] = {{0, 0}, {1, 1}};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            Y.setValue(i, j, Y_values[i][j]);
        }
    }

//////////////////////////////////////////////////////////////////
    // Get Number of Examples
    int m = X.cols;
    // std::cout << "Number of columns: " << m << "\n";

//////////////////////////////////////////////////////////////////
    // Calculate dZ2
    Matrix dZ2 = A2 - Y;
    
//////////////////////////////////////////////////////////////////
    // Calculate dW2
    // Transpose A1
    Matrix* A1_transpose = transpose_matrix(&A1);

    // Multiply dZ2 with A1.T
    Matrix dW2_temp = matmul(dZ2, *A1_transpose);

    // Divide by number of examples
    Matrix dW2 = dW2_temp / m;

//////////////////////////////////////////////////////////////////
    // Calculate db2
    Vector db2_temp = sum_columns(dZ2);

    // Divide by number of examples
    Vector db2 = db2_temp / m;

//////////////////////////////////////////////////////////////////
    // Calculate dZ1
    // Transpose W2 (next layer weights)
    Matrix* W_next_transpose = transpose_matrix(layer2->W);

    // Multiply W2.T with dZ2 (next layer dZ2)
    Matrix dZ1_temp = matmul(*W_next_transpose, dZ2);
    
    // Make a copy of Z1
    Matrix Z1_deriv(Z1.rows, Z1.cols);
    for (int i = 0; i < Z1.rows; i++) {
        for (int j = 0; j < Z1.cols; j++) {
            Z1_deriv.setValue(i, j, Z1.getValues(i, j));
        }
    }

    // Apply Sigmoid derivative to Z1_deriv
    sigmoid.derivative(Z1_deriv);

    // Element wise multiplication derivative activated Z1
    Matrix dZ1 = dZ1_temp * Z1_deriv;
   
//////////////////////////////////////////////////////////////////
    // Calculate dW1
    // Transpose X
    Matrix* X_T = transpose_matrix(&X);

    // Multiply current layer dZ (dZ1) with X.T
    Matrix dW1_temp = matmul(dZ1, *X_T);

    // Divide by number of examples
    Matrix dW1 = dW1_temp / m;

//////////////////////////////////////////////////////////////////
    // Calculate db1
    // Sum columns of dZ1
    Vector db1_temp = sum_columns(dZ1);

    // Divide by number of examples
    Vector db1 = db1_temp / m;

//////////////////////////////////////////////////////////////////
    // Test backward propagation function
    // Print divider
    std::cout << "________________________________" << std::endl;

    // Run backward propagation
    nn.backward(X,Y,"CatCrossEntropy");

//////////////////////////////////////////////////////////////////
    // Print divider
    std::cout << "________________________________" << std::endl;

    // Preview Layer 2 dZ (from inside the neural network)
    std::cout << "Layer 2 dZ:" << std::endl;
    preview_matrix(nn.layers[1]->dZ, 4);

    // Preview Layer 2 dZ (from the test)
    std::cout << "Layer 2 dZ (from the test):" << std::endl;
    preview_matrix(&dZ2, 4);

    // Check values
    for (int i = 0; i < nn.layers[1]->dZ->rows; i++) {
        for (int j = 0; j < nn.layers[1]->dZ->cols; j++) {
            ASSERT_NEAR(nn.layers[1]->dZ->getValues(i, j),
                        dZ2.getValues(i, j), 1e-5);
        }
    }

    // Preview Layer 2 dW (from inside the neural network)
    std::cout << "Layer 2 dW:" << std::endl;
    preview_matrix(nn.layers[1]->dW, 4);

    // Preview Layer 2 dW (from the test)
    std::cout << "Layer 2 dW (from the test):" << std::endl;
    preview_matrix(&dW2, 4);

    // Check values
    for (int i = 0; i < nn.layers[1]->dW->rows; i++) {
        for (int j = 0; j < nn.layers[1]->dW->cols; j++) {
            ASSERT_NEAR(nn.layers[1]->dW->getValues(i, j),
                        dW2.getValues(i, j), 1e-5);
        }
    }

    // Preview Layer 2 db (from inside the neural network)
    std::cout << "Layer 2 db:" << std::endl;
    preview_vector(nn.layers[1]->db, 4);

    // Preview Layer 2 db (from the test)
    std::cout << "Layer 2 db (from the test):" << std::endl;
    preview_vector(&db2, 4);

    // Check values
    for (int i = 0; i < nn.layers[1]->db->rows; i++) {
        ASSERT_NEAR(nn.layers[1]->db->getValues(i),
                    db2.getValues(i), 1e-5);
    }

    // Preview Layer 1 dZ (from inside the neural network)
    std::cout << "Layer 1 dZ:" << std::endl;
    preview_matrix(nn.layers[0]->dZ, 4);

    // Preview Layer 1 dZ (from the test)
    std::cout << "Layer 1 dZ (from the test):" << std::endl;
    preview_matrix(&dZ1, 4);

    // Check values
    for (int i = 0; i < nn.layers[0]->dZ->rows; i++) {
        for (int j = 0; j < nn.layers[0]->dZ->cols; j++) {
            ASSERT_NEAR(nn.layers[0]->dZ->getValues(i, j),
                        dZ1.getValues(i, j), 1e-5);
        }
    }

    // Preview Layer 1 dW (from inside the neural network)
    std::cout << "Layer 1 dW:" << std::endl;
    preview_matrix(nn.layers[0]->dW, 4);

    // Preview Layer 1 dW (from the test)
    std::cout << "Layer 1 dW (from the test):" << std::endl;
    preview_matrix(&dW1, 4);

    // Check values
    for (int i = 0; i < nn.layers[0]->dW->rows; i++) {
        for (int j = 0; j < nn.layers[0]->dW->cols; j++) {
            ASSERT_NEAR(nn.layers[0]->dW->getValues(i, j),
                        dW1.getValues(i, j), 1e-5);
        }
    }

    // Preview Layer 1 db (from inside the neural network)
    std::cout << "Layer 1 db:" << std::endl;
    preview_vector(nn.layers[0]->db, 4);

    // Preview Layer 1 db (from the test)
    std::cout << "Layer 1 db (from the test):" << std::endl;
    preview_vector(&db1, 4);

    // Check values
    for (int i = 0; i < nn.layers[0]->db->rows; i++) {
        ASSERT_NEAR(nn.layers[0]->db->getValues(i),
                    db1.getValues(i), 1e-5);
    }
}

TEST(NeuralNetworkTest, UpdateParamsTest) {
//////////////////////////////////////////////////////////////////
    // Forward Propagation Section
//////////////////////////////////////////////////////////////////
    // Initialize input matrix
    Matrix X(3, 2);  // 3 features, 2 examples
    // Initialize 2D array for inputs
    double X_values[3][2] = {{1, 4}, {2, 5}, {3, 6}};
    // Loop over rows
    for (int i = 0; i < 3; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            X.setValue(i, j, X_values[i][j]);
        }
    }

    // Preview input matrix
    std::cout << "Input Matrix X:" << std::endl;
    preview_matrix(&X, 2);

//////////////////////////////////////////////////////////////////
    // Create a simple neural network
    NeuralNetwork nn;
    Layer* layer1 = new Layer(3, 2, "Sigmoid");
    Layer* layer2 = new Layer(2, 2, "Softmax");

//////////////////////////////////////////////////////////////////
    // Initialize 2D arrays for weights
    double W1_values[2][3] = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    // Initialize 1D arrays for biases
    double b1_values[2] = {0.1, 0.2};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 3; j++) {
            layer1->W->setValue(i, j, W1_values[i][j]);
        }
        layer1->b->setValue(i, b1_values[i]);
    }

    // Preview layer 1 weights
    std::cout << "Layer 1 Weights:" << std::endl;
    preview_matrix(layer1->W, 4);

    // Preview layer 1 biases
    std::cout << "Layer 1 Biases:" << std::endl;
    preview_vector(layer1->b, 4);

//////////////////////////////////////////////////////////////////
    // Initialize 2D arrays for weights
    double W2_values[2][2] = {{0.7, 0.8}, {0.9, 1.0}};
    // Initialize 1D arrays for biases
    double b2_values[2] = {0.3, 0.4};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            layer2->W->setValue(i, j, W2_values[i][j]);
        }
        layer2->b->setValue(i, b2_values[i]);
    }

    // Preview layer 2 weights
    std::cout << "Layer 2 Weights:" << std::endl;
    preview_matrix(layer2->W, 4);

    // Preview layer 2 biases
    std::cout << "Layer 2 Biases:" << std::endl;
    preview_vector(layer2->b, 4);

//////////////////////////////////////////////////////////////////
    // Calculate Z1
    // Multiply weights with input
    Matrix Z1_temp = matmul(*layer1->W, X);
    // Initialize matrix for Z1
    Matrix Z1(Z1_temp.rows, Z1_temp.cols);
    // Loop over rows
    for (int i = 0; i < Z1.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z1.cols; j++) {
            // Add bias to the result
            Z1.setValue(i, j,
                        Z1_temp.getValues(i, j) + layer1->b->getValues(i));
        }
    }
    // Preview Z1
    std::cout << "Manual Z1:" << std::endl;
    preview_matrix(&Z1, 4);
    
//////////////////////////////////////////////////////////////////
    // Calculate A1
    Matrix A1(Z1.rows, Z1.cols);
    // Loop over rows
    for (int i = 0; i < Z1.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z1.cols; j++) {
            // Set values
            A1.setValue(i, j, Z1.getValues(i, j));
        }
    }
    // Initialize Sigmoid object
    Sigmoid sigmoid;
    // Apply Sigmoid function
    sigmoid.function(A1);

    // Preview Activated Z1
    std::cout << "Activated Z1:" << std::endl;
    preview_matrix(&A1, 4);

//////////////////////////////////////////////////////////////////
    // Calculate Z2
    // Multiply weights with A1
    Matrix Z2_temp = matmul(*layer2->W, A1);
    // Initialize matrix for Z2
    Matrix Z2(Z2_temp.rows, Z2_temp.cols);
    // Loop over rows
    for (int i = 0; i < Z2.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z2.cols; j++) {
            // Add bias to the result
            Z2.setValue(i, j,
                        Z2_temp.getValues(i, j) + layer2->b->getValues(i));
        }
    }

    // Preview Z2
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&Z2, 4);

//////////////////////////////////////////////////////////////////
    // Calculate A2
    Matrix A2(Z2.rows, Z2.cols);
    // Loop over rows
    for (int i = 0; i < Z2.rows; i++) {
        // Loop over columns
        for (int j = 0; j < Z2.cols; j++) {
            // Set values
            A2.setValue(i, j, Z2.getValues(i, j));
        }
    }

    // Initialize Softmax object
    Softmax softmax;
    // Apply Softmax function
    softmax.function(A2);

    // Preview A2
    std::cout << "Activated Z2 (final output):" << std::endl;
    preview_matrix(&A2, 4);

//////////////////////////////////////////////////////////////////
    // Test Forward Function
    // Add layers to the neural network
    nn.add_layer(layer1);
    nn.add_layer(layer2);

    // Run forward propagation
    nn.forward(X);

//////////////////////////////////////////////////////////////////
    // Check Z and A Matrices
    // Get Z1 from neural network
    Matrix* Z1_nn = nn.layers[0]->Z;
    
    // Preview Z1 from neural network
    std::cout << "Neural Network Z1:" << std::endl;
    preview_matrix(Z1_nn, 4);

    // Preview Z1 from Test
    std::cout << "Manual Z1:" << std::endl;
    preview_matrix(&Z1, 4);

    // Get A1 from neural network
    Matrix* A1_nn = nn.layers[0]->A;

    // Preview A1 from neural network
    std::cout << "Neural Network A1:" << std::endl;
    preview_matrix(A1_nn, 4);

    // Preview A1 from Test
    std::cout << "Manual A1:" << std::endl;
    preview_matrix(&A1, 4);

    // Get Z2 from neural network
    Matrix* Z2_nn = nn.layers[1]->Z;

    // Preview Z2 from neural network
    std::cout << "Neural Network Z2:" << std::endl;
    preview_matrix(Z2_nn, 4);

    // Preview Z2 from Test
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&Z2, 4);

    // Preview A2 from neural network
    Matrix* A2_nn = nn.layers[1]->A;

    // Preview A2 from neural network
    std::cout << "Neural Network A2:" << std::endl;
    preview_matrix(A2_nn, 4);

    // Preview A2 from Test
    std::cout << "Manual A2:" << std::endl;
    preview_matrix(&A2, 4);

//////////////////////////////////////////////////////////////////
    // Check output values
    // Get the output of the last layer
    Matrix* output = nn.getOutput();

    // Preview Neural Network output
    std::cout << "Neural Network output:" << std::endl;
    preview_matrix(output, 4);

    // Preview output from test
    std::cout << "Manual Z2:" << std::endl;
    preview_matrix(&A2, 4);

    // Check values
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++) {
            ASSERT_NEAR(output->getValues(i, j),
                        A2.getValues(i, j), 1e-5);
        }
    }

//////////////////////////////////////////////////////////////////
    // Backward Propagation Section
    // Print divider
    std::cout << "________________________________" << std::endl;
//////////////////////////////////////////////////////////////////
    // Initialize output matrix
    Matrix Y(2, 2);  // 2 outputs, 2 examples
    // Initialize 2D array for inputs
    double Y_values[3][2] = {{0, 0}, {1, 1}};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            Y.setValue(i, j, Y_values[i][j]);
        }
    }

//////////////////////////////////////////////////////////////////
    // Get Number of Examples
    int m = X.cols;
    // std::cout << "Number of columns: " << m << "\n";

//////////////////////////////////////////////////////////////////
    // Calculate dZ2
    Matrix dZ2 = A2 - Y;
    
//////////////////////////////////////////////////////////////////
    // Calculate dW2
    // Transpose A1
    Matrix* A1_transpose = transpose_matrix(&A1);

    // Multiply dZ2 with A1.T
    Matrix dW2_temp = matmul(dZ2, *A1_transpose);

    // Divide by number of examples
    Matrix dW2 = dW2_temp / m;

//////////////////////////////////////////////////////////////////
    // Calculate db2
    Vector db2_temp = sum_columns(dZ2);

    // Divide by number of examples
    Vector db2 = db2_temp / m;

//////////////////////////////////////////////////////////////////
    // Calculate dZ1
    // Transpose W2 (next layer weights)
    Matrix* W_next_transpose = transpose_matrix(layer2->W);

    // Multiply W2.T with dZ2 (next layer dZ2)
    Matrix dZ1_temp = matmul(*W_next_transpose, dZ2);
    
    // Make a copy of Z1
    Matrix Z1_deriv(Z1.rows, Z1.cols);
    for (int i = 0; i < Z1.rows; i++) {
        for (int j = 0; j < Z1.cols; j++) {
            Z1_deriv.setValue(i, j, Z1.getValues(i, j));
        }
    }

    // Apply Sigmoid derivative to Z1_deriv
    sigmoid.derivative(Z1_deriv);

    // Element wise multiplication derivative activated Z1
    Matrix dZ1 = dZ1_temp * Z1_deriv;
   
//////////////////////////////////////////////////////////////////
    // Calculate dW1
    // Transpose X
    Matrix* X_T = transpose_matrix(&X);

    // Multiply current layer dZ (dZ1) with X.T
    Matrix dW1_temp = matmul(dZ1, *X_T);

    // Divide by number of examples
    Matrix dW1 = dW1_temp / m;

//////////////////////////////////////////////////////////////////
    // Calculate db1
    // Sum columns of dZ1
    Vector db1_temp = sum_columns(dZ1);

    // Divide by number of examples
    Vector db1 = db1_temp / m;

//////////////////////////////////////////////////////////////////
    // Test backward propagation function
    // Print divider
    std::cout << "________________________________" << std::endl;

    // Run backward propagation
    nn.backward(X,Y,"CatCrossEntropy");

//////////////////////////////////////////////////////////////////
    // Print divider
    std::cout << "________________________________" << std::endl;

    // Preview Layer 2 dZ (from inside the neural network)
    std::cout << "Layer 2 dZ:" << std::endl;
    preview_matrix(nn.layers[1]->dZ, 4);

    // Preview Layer 2 dZ (from the test)
    std::cout << "Layer 2 dZ (from the test):" << std::endl;
    preview_matrix(&dZ2, 4);

    // Check values
    for (int i = 0; i < nn.layers[1]->dZ->rows; i++) {
        for (int j = 0; j < nn.layers[1]->dZ->cols; j++) {
            ASSERT_NEAR(nn.layers[1]->dZ->getValues(i, j),
                        dZ2.getValues(i, j), 1e-5);
        }
    }

    // Preview Layer 2 dW (from inside the neural network)
    std::cout << "Layer 2 dW:" << std::endl;
    preview_matrix(nn.layers[1]->dW, 4);

    // Preview Layer 2 dW (from the test)
    std::cout << "Layer 2 dW (from the test):" << std::endl;
    preview_matrix(&dW2, 4);

    // Check values
    for (int i = 0; i < nn.layers[1]->dW->rows; i++) {
        for (int j = 0; j < nn.layers[1]->dW->cols; j++) {
            ASSERT_NEAR(nn.layers[1]->dW->getValues(i, j),
                        dW2.getValues(i, j), 1e-5);
        }
    }

    // Preview Layer 2 db (from inside the neural network)
    std::cout << "Layer 2 db:" << std::endl;
    preview_vector(nn.layers[1]->db, 4);

    // Preview Layer 2 db (from the test)
    std::cout << "Layer 2 db (from the test):" << std::endl;
    preview_vector(&db2, 4);

    // Check values
    for (int i = 0; i < nn.layers[1]->db->rows; i++) {
        ASSERT_NEAR(nn.layers[1]->db->getValues(i),
                    db2.getValues(i), 1e-5);
    }

    // Preview Layer 1 dZ (from inside the neural network)
    std::cout << "Layer 1 dZ:" << std::endl;
    preview_matrix(nn.layers[0]->dZ, 4);

    // Preview Layer 1 dZ (from the test)
    std::cout << "Layer 1 dZ (from the test):" << std::endl;
    preview_matrix(&dZ1, 4);

    // Check values
    for (int i = 0; i < nn.layers[0]->dZ->rows; i++) {
        for (int j = 0; j < nn.layers[0]->dZ->cols; j++) {
            ASSERT_NEAR(nn.layers[0]->dZ->getValues(i, j),
                        dZ1.getValues(i, j), 1e-5);
        }
    }

    // Preview Layer 1 dW (from inside the neural network)
    std::cout << "Layer 1 dW:" << std::endl;
    preview_matrix(nn.layers[0]->dW, 4);

    // Preview Layer 1 dW (from the test)
    std::cout << "Layer 1 dW (from the test):" << std::endl;
    preview_matrix(&dW1, 4);

    // Check values
    for (int i = 0; i < nn.layers[0]->dW->rows; i++) {
        for (int j = 0; j < nn.layers[0]->dW->cols; j++) {
            ASSERT_NEAR(nn.layers[0]->dW->getValues(i, j),
                        dW1.getValues(i, j), 1e-5);
        }
    }

    // Preview Layer 1 db (from inside the neural network)
    std::cout << "Layer 1 db:" << std::endl;
    preview_vector(nn.layers[0]->db, 4);

    // Preview Layer 1 db (from the test)
    std::cout << "Layer 1 db (from the test):" << std::endl;
    preview_vector(&db1, 4);

    // Check values
    for (int i = 0; i < nn.layers[0]->db->rows; i++) {
        ASSERT_NEAR(nn.layers[0]->db->getValues(i),
                    db1.getValues(i), 1e-5);
    }
//////////////////////////////////////////////////////////////////
    // Update Parameters Section
    double learning_rate = 0.01;

    // Create matrix for W2_test
    Matrix W2_test(layer2->W->rows, layer2->W->cols);
    for (int i = 0; i < layer2->W->rows; i++) {
        for (int j = 0; j < layer2->W->cols; j++) {
            W2_test.setValue(i, j, layer2->W->getValues(i, j));
        }
    }

    // Create matrix for b2_test
    Vector b2_test(layer2->b->rows);
    for (int i = 0; i < layer2->b->rows; i++) {
        b2_test.setValue(i, layer2->b->getValues(i));
    }

    // Create matrix for W1_test
    Matrix W1_test(layer1->W->rows, layer1->W->cols);
    for (int i = 0; i < layer1->W->rows; i++) {
        for (int j = 0; j < layer1->W->cols; j++) {
            W1_test.setValue(i, j, layer1->W->getValues(i, j));
        }
    }

    // Create matrix for b1_test
    Vector b1_test(layer1->b->rows);
    for (int i = 0; i < layer1->b->rows; i++) {
        b1_test.setValue(i, layer1->b->getValues(i));
    }

//////////////////////////////////////////////////////////////////
    // Update W2
    for (int i = 0; i < layer2->W->rows; i++) {
        for (int j = 0; j < layer2->W->cols; j++) {
            // Get current weight and gradient
            double current_weight = layer2->W->getValues(i, j);
            double current_gradient = dW2.getValues(i, j);

            // Multiply gradient by learning rate
            double weight_delta = learning_rate * current_gradient;

            // Update weight
            double update_weight = current_weight - weight_delta;

            // Set new weight
            W2_test.setValue(i, j, update_weight);
        }
    }

//////////////////////////////////////////////////////////////////
    // Update b2
    for (int i = 0; i < layer2->b->rows; i++) {
        // Get current bias and gradient
        double current_bias = layer2->b->getValues(i);
        double gradient_bias = db2.getValues(i);

        // Multiply gradient by learning rate
        double bias_delta = learning_rate * gradient_bias;

        // Update bias
        double update_bias = current_bias - bias_delta;

        // Set new bias
        b2_test.setValue(i, update_bias);
    }

//////////////////////////////////////////////////////////////////
    // Update W1
    for (int i = 0; i < layer1->W->rows; i++) {
        for (int j = 0; j < layer1->W->cols; j++) {
            // Get current weight and gradient
            double current_weight = layer1->W->getValues(i, j);
            double current_gradient = dW1.getValues(i, j);

            // Multiply gradient by learning rate
            double weight_delta = learning_rate * current_gradient;

            // Update weight
            double update_weight = current_weight - weight_delta;

            // Set new weight
            W1_test.setValue(i, j, update_weight);
        }
    }

//////////////////////////////////////////////////////////////////
    // Update b1
    for (int i = 0; i < layer1->b->rows; i++) {
        // Get current bias and gradient
        double current_bias = layer1->b->getValues(i);
        double gradient_bias = db1.getValues(i);

        // Multiply gradient by learning rate
        double bias_delta = learning_rate * gradient_bias;

        // Update bias
        double update_bias = current_bias - bias_delta;

        // Set new bias
        b1_test.setValue(i, update_bias);
    }

//////////////////////////////////////////////////////////////////
// Run the Update Parameters function
    nn.update_params(learning_rate);

//////////////////////////////////////////////////////////////////
    // Test update parameters functions
    // Print divider
    std::cout << "________________________________" << std::endl;

    // Preview W2_test
    std::cout << "W2_test:" << std::endl;
    preview_matrix(&W2_test, 4);

    // Preview W2 from neural network
    std::cout << "Neural Network W2:" << std::endl;
    preview_matrix(nn.layers[1]->W, 4);

    // Check values
    for (int i = 0; i < nn.layers[1]->W->rows; i++) {
        for (int j = 0; j < nn.layers[1]->W->cols; j++) {
            ASSERT_NEAR(nn.layers[1]->W->getValues(i, j),
                        W2_test.getValues(i, j), 1e-5);
        }
    }

    // Preview b2_test
    std::cout << "b2_test:" << std::endl;
    preview_vector(&b2_test, 4);

    // Preview b2 from neural network
    std::cout << "Neural Network b2:" << std::endl;
    preview_vector(nn.layers[1]->b, 4);

    // Check values
    for (int i = 0; i < nn.layers[1]->b->rows; i++) {
        ASSERT_NEAR(nn.layers[1]->b->getValues(i),
                    b2_test.getValues(i), 1e-5);
    }

    // Preview W1_test
    std::cout << "W1_test:" << std::endl;
    preview_matrix(&W1_test, 4);

    // Preview W1 from neural network
    std::cout << "Neural Network W1:" << std::endl;
    preview_matrix(nn.layers[0]->W, 4);

    // Check values
    for (int i = 0; i < nn.layers[0]->W->rows; i++) {
        for (int j = 0; j < nn.layers[0]->W->cols; j++) {
            ASSERT_NEAR(nn.layers[0]->W->getValues(i, j),
                        W1_test.getValues(i, j), 1e-5);
        }
    }

    // Preview b1_test
    std::cout << "b1_test:" << std::endl;
    preview_vector(&b1_test, 4);

    // Preview b1 from neural network
    std::cout << "Neural Network b1:" << std::endl;
    preview_vector(nn.layers[0]->b, 4);

    // Check values
    for (int i = 0; i < nn.layers[0]->b->rows; i++) {
        ASSERT_NEAR(nn.layers[0]->b->getValues(i),
                    b1_test.getValues(i), 1e-5);
    }
}
//////////////////////////////////////////////////////////////////
TEST(NeuralNetworkTest, GetAccuracyTest) {
    // Initialize input matrix
    Matrix X(3, 2);  // 3 features, 2 examples
    // Initialize 2D array for inputs
    double X_values[3][2] = {{1, 4}, {2, 5}, {3, 6}};
    // Loop over rows
    for (int i = 0; i < 3; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            X.setValue(i, j, X_values[i][j]);
        }
    }

    // Initialize output matrix
    Matrix Y(2, 2);  // 2 outputs, 2 examples
    // Initialize 2D array for inputs
    double Y_values[3][2] = {{0, 0}, {1, 1}};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            Y.setValue(i, j, Y_values[i][j]);
        }
    }

    // Create a simple neural network
    NeuralNetwork nn;
    Layer* layer1 = new Layer(3, 2, "Sigmoid");
    Layer* layer2 = new Layer(2, 2, "Softmax");

    // Initialize 2D arrays for weights
    double W1_values[2][3] = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    // Initialize 1D arrays for biases
    double b1_values[2] = {0.1, 0.2};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 3; j++) {
            layer1->W->setValue(i, j, W1_values[i][j]);
        }
        layer1->b->setValue(i, b1_values[i]);
    }

    // Initialize 2D arrays for weights
    double W2_values[2][2] = {{0.7, 0.8}, {0.9, 1.0}};
    // Initialize 1D arrays for biases
    double b2_values[2] = {0.3, 0.4};
    // Loop over rows
    for (int i = 0; i < 2; i++) {
        // Loop over columns
        for (int j = 0; j < 2; j++) {
            layer2->W->setValue(i, j, W2_values[i][j]);
        }
        layer2->b->setValue(i, b2_values[i]);
    }

    // Add layers to the neural network
    nn.add_layer(layer1);
    nn.add_layer(layer2);

    // Run forward propagation
    nn.forward(X);

    // Run backward propagation
    nn.backward(X,Y,"CatCrossEntropy");

    // Update Parameters
    double learning_rate = 0.01;
    nn.update_params(learning_rate);

    // Print divider
    std::cout << "________________________________" << std::endl;

    // Get output
    Matrix* Y_pred = nn.getOutput();

    // Preview Y
    std::cout << "Y:" << std::endl;
    preview_matrix(&Y, 4);

    // Preview Y_pred
    std::cout << "Y_pred:" << std::endl;
    preview_matrix(Y_pred, 4);

    // Get argmax of true and predicted labels
    Vector true_labels = argmax(Y);
    Vector pred_labels = argmax(*Y_pred);

    // Preview true labels
    std::cout << "True Labels:" << std::endl;
    preview_vector(&true_labels, 4);

    // Preview predicted labels
    std::cout << "Predicted Labels:" << std::endl;
    preview_vector(&pred_labels, 4);

    // Check number of correct predictions
    double correct_count = 0.0;
    for (int i = 0; i < Y.cols; i++) {
        if (true_labels.getValues(i) == pred_labels.getValues(i)) {
            correct_count += 1.0;
        }
    }

    double test_accuracy = correct_count / Y.cols;

    // Get accuracy
    double accuracy = nn.get_accuracy(Y);

    // Print accuracy from test
    std::cout << "Test Accuracy: " << test_accuracy << std::endl;

    // Print accuracy from function
    std::cout << "Function Accuracy: " << accuracy << std::endl;

    // Check values
    ASSERT_NEAR(accuracy, test_accuracy, 1e-5);

}

////////////////////////////////////////////////////////////////////
// Main Function
int main(int argc, char **argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    // Run all tests
    return RUN_ALL_TESTS();
}
