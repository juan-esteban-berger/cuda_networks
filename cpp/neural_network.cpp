#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include "linear_algebra.h"
#include "neural_network.h"

//////////////////////////////////////////////////////////////////
// Activation Function Classes
void Sigmoid::function(Matrix& Z) {
    for (int i = 0; i < Z.rows; i++) {
        for (int j = 0; j < Z.cols; j++) {
            double z = Z.getValues(i, j);
            Z.setValue(i, j, 1.0 / (1.0 + std::exp(-z)));
        }
    }
}

void Sigmoid::derivative(Matrix& Z) {
    for (int i = 0; i < Z.rows; i++) {
        for (int j = 0; j < Z.cols; j++) {
            double z = Z.getValues(i, j);
            double sigmoid_z = 1.0 / (1.0 + std::exp(-z));
            Z.setValue(i, j, sigmoid_z * (1.0 - sigmoid_z));
        }
    }
}

void Softmax::function(Matrix& Z) {
    // Loop through each column
    for (int j = 0; j < Z.cols; j++) {
        // Find max value in the column
        double max_val = Z.getValues(0, j);
        for (int i = 1; i < Z.rows; i++) {
            double temp_val = Z.getValues(i, j);
            if (temp_val > max_val) {
                max_val = temp_val;
            }
        }

        // Compute exp(Z - max)
        double sum = 0.0;
        for (int i = 0; i < Z.rows; i++) {
            double exp_val = std::exp(Z.getValues(i, j) - max_val);
            Z.setValue(i, j, exp_val);
        }

        // Compute sum(exp(Z - max))
        for (int i = 0; i < Z.rows; i++) {
            sum += Z.getValues(i, j);
        }

        // Divide by sum
        for (int i = 0; i < Z.rows; i++) {
            Z.setValue(i, j, Z.getValues(i, j) / (sum + 1e-8));
        }
    }
}

//////////////////////////////////////////////////////////////////
// Loss Function Classes
double CatCrossEntropy::function(Matrix& Y, Matrix& Y_hat) {
    double loss = 0.0;
    for (int i = 0; i < Y.rows; i++) {
        for (int j = 0; j < Y.cols; j++) {
            double y = Y.getValues(i, j);
            double y_hat = Y_hat.getValues(i, j);
            loss -= y * log(y_hat + 1e-8);
        }
    }
    return loss;
}

//////////////////////////////////////////////////////////////////
// Layer Class
Layer::Layer(int input_num, int output_num, std::string activation_func) {
    activation = activation_func;
    W = new Matrix(output_num, input_num);
    b = new Vector(output_num);
    
    random_matrix(W);
    random_vector(b);

    Z = nullptr;
    A = nullptr;
    dZ = nullptr;
    dW = nullptr;
    db = nullptr;
}

Layer::~Layer() {
    delete W;
    delete b;
    delete Z;
    delete A;
    delete dZ;
    delete dW;
    delete db;
}

//////////////////////////////////////////////////////////////////
// Neural Network Class
NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::~NeuralNetwork() {
    for (Layer* layer : layers) {
        delete layer;
    }
}

void NeuralNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
}

Matrix* NeuralNetwork::getOutput() {
    return layers.back()->A;
}

//////////////////////////////////////////////////////////////////
void NeuralNetwork::forward(Matrix& X) {
//////////////////////////////////////////////////////////////////
    // Initialize Matrix A
    Matrix A(X.rows, X.cols);

    // Copy from Matrix X into Matrix A
    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            A.setValue(i, j, X.getValues(i, j));
        }
    }

//////////////////////////////////////////////////////////////////
    // Initialize Sigmoid object
    Sigmoid sigmoid;
    // Initialize Softmax object
    Softmax softmax;
    
//////////////////////////////////////////////////////////////////
    // Loop through each layer
    for (Layer* layer : layers) {
//////////////////////////////////////////////////////////////////
        // Calculate Z
        // Multiply weights by input matrix
        Matrix Z_temp = matmul(*layer->W, A);

        // Add bias to Z matrix
        Matrix Z = Z_temp + *layer->b;
        
        // Create a new Matrix for Z and copy values
        layer->Z = new Matrix(Z.rows, Z.cols);
        for (int i = 0; i < Z.rows; i++) {
            for (int j = 0; j < Z.cols; j++) {
                layer->Z->setValue(i, j, Z.getValues(i, j));
            }
        }
        
//////////////////////////////////////////////////////////////////
        // Calculate A
        // if (layer->activation == "Sigmoid")
        if (layer->activation == "Sigmoid") {
            // Compute Sigmoid function
            sigmoid.function(Z);

        } else if (layer->activation == "Softmax") {
            // Compute Softmax function
            softmax.function(Z);

        }

        // Create a new matrix for A copy values
        layer->A = new Matrix(Z.rows, Z.cols);
        for (int i = 0; i < Z.rows; i++) {
            for (int j = 0; j < Z.cols; j++) {
                layer->A->setValue(i, j, Z.getValues(i, j));
            }
        }

//////////////////////////////////////////////////////////////////
        // Update A for the next iteration
        Matrix A_temp(layer->A->rows, layer->A->cols);
        for (int i = 0; i < layer->A->rows; i++) {
            for (int j = 0; j < layer->A->cols; j++) {
                A_temp.setValue(i, j, layer->A->getValues(i, j));
            }
        }

        // Deallocate Memory
        for (int i = 0; i < A.rows; i++) {
            delete[] A.data[i];
        }
        delete[] A.data;

        // Deallocate Memory
        A.rows = A_temp.rows;
        A.cols = A_temp.cols;
        A.data = new double*[A.rows];

        // Copy Values
        for (int i = 0; i < A.rows; i++) {
            A.data[i] = new double[A.cols];
            for (int j = 0; j < A.cols; j++) {
                A.setValue(i, j, A_temp.getValues(i, j));
            }
        }

    }
}

//////////////////////////////////////////////////////////////////
void NeuralNetwork::backward(Matrix& X,
                             Matrix& Y,
                             std::string loss_func) {
//////////////////////////////////////////////////////////////////
// Function Setup
    // Get Number of Examples
    int m = X.cols;
    // std::cout << "Number of Columns: " << m << std::endl;

    // Initialize Sigmoid Object
    Sigmoid sigmoid;
    
//////////////////////////////////////////////////////////////////
    // Iterate through each layer reverse
    for (int i = layers.size() - 1; i >= 0; i--) {

//////////////////////////////////////////////////////////////////
// Initialize matrices for gradients
        // Initialize a Matrix for current layer's dZ
        layers[i]->dZ = new Matrix(layers[i]->A->rows,
                                   layers[i]->A->cols);

        // Initialize a Matrix for current layer's dW
        layers[i]->dW = new Matrix(layers[i]->W->rows,
                                   layers[i]->W->cols);

        // Initialize a Vector for current layer's db
        layers[i]->db = new Vector(layers[i]->b->rows);

//////////////////////////////////////////////////////////////////
// Initialize pointer to current layer
        // Pointer to layer
        Layer* layer = layers[i];

//////////////////////////////////////////////////////////////////
// Calculate dZ
        // Calculate dZ: if last layer and loss is CatCrossEntropy
        if (i == layers.size() - 1 && loss_func == "CatCrossEntropy") {
            // Matrix dZ_temp
            Matrix dZ_temp = *layers[i]->A - Y;

            // Copy the values to layer's dZ using the pointer to layer
            for (int i = 0; i < dZ_temp.rows; i++) {
                for (int j = 0; j < dZ_temp.cols; j++) {
                    layer->dZ->setValue(i, j, dZ_temp.getValues(i, j));
                }
            }

        }
        // Calculate dZ: all other layers
        else {
            // Get the next layer's dZ
            Matrix* dZ_next = layers[i + 1]->dZ;
            
            // Get the next layer's W
            Matrix* W_next = layers[i + 1]->W;

            // Transpose Matrix W_next
            Matrix* W_next_transpose = transpose_matrix(W_next);

            // Multiply W_next.T with dZ_next
            Matrix dZ_temp = matmul(*W_next_transpose, *dZ_next);

            // Make a copy of pervious layers Z
            Matrix Z_deriv(layers[i]->Z->rows, layers[i]->Z->cols);
            for (int row = 0; row < layers[i]->Z->rows; row++) {
                for (int col = 0; col < layers[i]->Z->cols; col++) {
                    Z_deriv.setValue(row, col,
                    layers[i]->Z->getValues(row, col));
                }
            }

            if (layers[i]->activation == "Sigmoid") {
                // Compute Sigmoid derivative
                sigmoid.derivative(Z_deriv);
            }

            Matrix dZ = dZ_temp * Z_deriv;

            // Copy the values to layer's dZ using the pointer to layer
            for (int i = 0; i < dZ.rows; i++) {
                for (int j = 0; j < dZ.cols; j++) {
                    layer->dZ->setValue(i, j, dZ.getValues(i, j));
                }
            }
            
        }

//////////////////////////////////////////////////////////////////
// Calculate dW
        // Calculate dW: all except the first layer
        if (i != 0) {
            // Transpose Matrix Previous Layer's A
            Matrix* A_transpose = transpose_matrix(layers[i - 1]->A);

            // Multiply dZ with A.T
            Matrix dW_temp = matmul(*layers[i]->dZ, *A_transpose);

            // Divide by scalar (operator overload for /)
            Matrix dW = dW_temp / m;

            // Copy the values to layer's dW using the pointer to layer
            for (int i = 0; i < dW.rows; i++) {
                for (int j = 0; j < dW.cols; j++) {
                    layer->dW->setValue(i, j, dW.getValues(i, j));
                }
            }
        }
        // Calculate dW: first layer
        else {
            // Transpose Matrix X
            Matrix* X_transpose = transpose_matrix(&X);

            // Multiply dZ with X.T
            Matrix dW_temp = matmul(*layers[i]->dZ, *X_transpose);

            // Divide by scalar (operator overload for /)
            Matrix dW = dW_temp / m;

            // Copy the values to layer's dW using the pointer to layer
            for (int i = 0; i < dW.rows; i++) {
                for (int j = 0; j < dW.cols; j++) {
                    layer->dW->setValue(i, j, dW.getValues(i, j));
                }
            }
        }

//////////////////////////////////////////////////////////////////
// Calculate db
        // Sum columns of dZ
        Vector db_temp = sum_columns(*layers[i]->dZ);

        // Divide by scalar (operator overload for /)
        Vector db = db_temp / m;

        // Copy the values to layer's db using the pointer to layer
        for (int i = 0; i < db.rows; i++) {
            layer->db->setValue(i, db.getValues(i));
        }
    }
}

//////////////////////////////////////////////////////////////////
void NeuralNetwork::update_params(double learning_rate) {
    // Iterate through each layer using index
    for (int idx = 0; idx < layers.size(); idx++) {
        Layer* layer = layers[idx];

        // Update weights
        for (int i = 0; i < layer->W->rows; i++) {
            for (int j = 0; j < layer->W->cols; j++) {
                // Get current weight and gradient
                double current_weight = layer->W->getValues(i, j);
                double gradient_weight = layer->dW->getValues(i, j);

                // Multiply gradient by learning rate
                double weight_delta = learning_rate * gradient_weight;

                // Update weight
                double updated_weight = current_weight - weight_delta;

                // Set new weight
                layer->W->setValue(i, j, updated_weight);
            }
        }

        // Update biases
        for (int i = 0; i < layer->b->rows; i++) {
            // Get current bias and gradient
            double current_bias = layer->b->getValues(i);
            double gradient_bias = layer->db->getValues(i);

            // Multiply gradient by learning rate
            double bias_delta = learning_rate * gradient_bias;

            // Update bias
            double updated_bias = current_bias - bias_delta;

            // Set new bias
            layer->b->setValue(i, updated_bias);
        }
    }
}

//////////////////////////////////////////////////////////////////
double NeuralNetwork::get_accuracy(Matrix& Y_true) {
    // Get output
    Matrix* Y_pred = getOutput();

    // Preview true labels
    std::cout << "True Labels: \n";
    preview_matrix(&Y_true, 4);

    // Preview output
    std::cout << "Output: \n";
    preview_matrix(Y_pred, 4);

    // Get argmax of true and predicted labels
    Vector true_labels = argmax(Y_true);
    Vector pred_labels = argmax(*Y_pred);

    // Preview true and predicted labels
    std::cout << "True Labels: \n";
    preview_vector(&true_labels, 4);

    std::cout << "Predicted Labels: \n";
    preview_vector(&pred_labels, 4);

    // Check number of correct predictions
    double correct_count = 0.0;
    for (int i = 0; i < Y_true.cols; ++i) {
        if (pred_labels.getValues(i) == true_labels.getValues(i)) {
            correct_count += 1.0;
        }
    }

    return correct_count / Y_true.cols;
}

void NeuralNetwork::train(Matrix& X_train,
                          Matrix& Y_train,
                          int epochs,
                          double learning_rate,
                          std::string loss,
                          std::string history_path) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Print accuracy
        std::cout << "Epoch:" << epoch << "\n";

        std::cout << "Forward Pass\n";
        forward(X_train);
        
        // // Preview output
        // Matrix* output = getOutput();
        // std::cout << "Output: \n";
        // preview_matrix(output, 4);

        std::cout << "Backward Pass\n";
        backward(X_train, Y_train, loss);

        // Preview Gradients
        std::cout << "Layer 2 dZ: \n";
        preview_matrix(layers[1]->dZ, 4);

        std::cout << "Layer 2 dW: \n";
        preview_matrix(layers[1]->dW, 4);

        std::cout << "Layer 2 db: \n";
        preview_vector(layers[1]->db, 4);

        std::cout << "Update Parameters\n";
        update_params(learning_rate);

        // Print accuracy
        double accuracy = get_accuracy(Y_train);
        std::cout << "Accuracy: " << accuracy << "\n";
    }
}
