#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <ctime>
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

void NeuralNetwork::describe() const {
    std::cout << "Neural Network Architecture:" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i + 1 << ":" << std::endl;
        std::cout << "  Input:      " << layers[i]->W->cols << std::endl;
        std::cout << "  Output:     " << layers[i]->W->rows << std::endl;
        std::cout << "  Activation: " << layers[i]->activation << std::endl;
        std::cout << std::endl;
    }
}

void NeuralNetwork::preview_parameters(int decimals) const {
    std::cout << "Neural Network Parameters:" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i + 1 << ":" << std::endl;
        
        std::cout << "Weights (W):" << std::endl;
        preview_matrix(layers[i]->W, decimals);
        
        std::cout << "Biases (b):" << std::endl;
        preview_vector(layers[i]->b, decimals);
        
        std::cout << std::endl;
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
double NeuralNetwork::get_accuracy(Matrix& X, Matrix& Y_true) {
    forward(X);
    Matrix* Y_pred = getOutput();
    Vector true_labels = argmax(Y_true);
    Vector pred_labels = argmax(*Y_pred);
    double correct_count = 0.0;
    for (int i = 0; i < Y_true.cols; ++i) {
        if (pred_labels.getValues(i) == true_labels.getValues(i)) {
            correct_count += 1.0;
        }
    }
    return correct_count / Y_true.cols;
}

void progress_bar(int epoch,
                  int total_epochs,
                  double accuracy,
                  double loss,
                  double duration) {
    // Calculate percentage of completion
    double progress = (epoch + 1) / static_cast<double>(total_epochs);

    // Define the width for the progress bar
    int barWidth = 25;
    int pos = static_cast<int>(barWidth * progress);

    // Clear the current line
    std::cout << "\r"; // Carriage return to the beginning of the line
    std::cout << "Epoch: " << std::setw(3) << epoch + 1 << "/" << total_epochs
              << ", Accuracy: " << std::fixed << std::setprecision(3) << accuracy
              << ", Loss: " << std::fixed << std::setprecision(0) << loss << " "
              << ", Duration: " << std::fixed << std::setprecision(4) << duration << "s "
              << std::setw(3) << int(progress * 100.0) << "% [";

    // Draw the progress bar
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "]";

    // Flush the stream to ensure immediate display
    std::cout.flush();

    // If this is the last epoch, print a new line
    if (epoch == total_epochs - 1) {
        std::cout << std::endl;
    }
}

void save_history(std::string history_path,
                  std::vector<int>& epoch_list,
                  std::vector<double>& accuracy_list,
                  std::vector<double>& loss_list,
                  std::vector<double>& duration_list) {
    // Open the file for writing
    std::ofstream file(history_path);

    // Write the data
    for (int i = 0; i < epoch_list.size(); ++i) {
        file << epoch_list[i] << ","
             << accuracy_list[i] << ","
             << loss_list[i] << ","
             << duration_list[i] << std::endl;
    }

    // Close the file
    file.close();
}

void NeuralNetwork::gradient_descent(Matrix& X_train,
                                     Matrix& Y_train,
                                     std::string loss,
                                     double learning_rate) {
    forward(X_train);
    backward(X_train, Y_train, loss);
    update_params(learning_rate);
}

void NeuralNetwork::train(Matrix& X_train,
                          Matrix& Y_train,
                          int epochs,
                          double learning_rate,
                          std::string loss,
                          std::string optimizer,
                          double batch_size,
                          std::string history_path) {

    // Vectors to store history
    std::vector<int> epoch_list;
    std::vector<double> accuracy_list;
    std::vector<double> loss_list;
    std::vector<double> duration_list;

    // Start the timer
    std::clock_t start_time = std::clock();
    
    // Iterate through epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Initialize variables
        double acc = 0.0;
        double loss_val = 0.0;

        if (optimizer == "batch_gradient_descent") {
            // Gradient Descent
            gradient_descent(X_train, Y_train, loss, learning_rate);
            // Calculate accuracy
            acc = get_accuracy(X_train, Y_train);
            // Calculate loss
            double loss_val = 0.0;
            if (loss == "CatCrossEntropy") {
                CatCrossEntropy ce;
                loss_val = ce.function(Y_train, *getOutput());
            }
        } else if (optimizer == "mini_batch_gradient_descent") {
            // Temporary lists
            std::vector<double> acc_temp_list;
            std::vector<double> loss_temp_list;
            // Loop through mini-batches
            for (int i = 0; i < X_train.cols; i += batch_size) {
                // Get end index
                int end_idx = i + batch_size;
                // Check if end index is greater than total columns
                if (end_idx > X_train.cols) end_idx = X_train.cols;

                // Get mini-batches
                Matrix X_batch = X_train.iloc(0, X_train.rows, i, end_idx);
                Matrix Y_batch = Y_train.iloc(0, Y_train.rows, i, end_idx);
               
                // Gradient Descent
                gradient_descent(X_batch, Y_batch, loss, learning_rate);
                
                // Calculate accuracy
                double acc_batch = get_accuracy(X_batch, Y_batch);
                acc_temp_list.push_back(acc_batch);
                
                if (loss == "CatCrossEntropy") {
                    CatCrossEntropy ce;
                    double loss_batch = ce.function(Y_batch, *getOutput());
                    loss_temp_list.push_back(loss_batch);
                }
            }
            // Calculate average accuracy and loss
            acc = 0.0;
            loss_val = 0.0;
            for (int i = 0; i < acc_temp_list.size(); ++i) {
                acc += acc_temp_list[i];
                loss_val += loss_temp_list[i];
            }
            acc /= acc_temp_list.size();
            loss_val /= loss_temp_list.size();
        }

        // Calculate duration
        std::clock_t end_time = std::clock();
        double duration = (end_time - start_time) / static_cast<double>(CLOCKS_PER_SEC);

        // Calculate accuracy
        double accuracy = get_accuracy(X_train, Y_train);

        // Append to history lists
        epoch_list.push_back(epoch);
        accuracy_list.push_back(accuracy);
        loss_list.push_back(loss_val);
        duration_list.push_back(duration);

        // Print progress bar
        progress_bar(epoch, epochs, accuracy, loss_val, duration);
    }

    // Save history
    save_history(history_path,
                 epoch_list,
                 accuracy_list,
                 loss_list,
                 duration_list);
}

Vector NeuralNetwork::predict(Matrix& X) {
    // Forward pass
    forward(X);
    // Get predictions
    Matrix* output = getOutput();
    // Get argmax of predictions
    Vector predictions = argmax(*output);

    return predictions;
}

void NeuralNetwork::save_config(std::string filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for writing: " << filepath << std::endl;
        return;
    }

    for (Layer* layer : layers) {
        int input_num = layer->W->cols;
        int output_num = layer->W->rows;
        std::string activation = layer->activation;
        file << input_num << "," << output_num << "," << activation << "\n";
    }

    file.close();
}

void NeuralNetwork::save_weights(std::string filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for writing: " << filepath << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(8);

    for (Layer* layer : layers) {
        // Save weights
        for (int i = 0; i < layer->W->rows; ++i) {
            for (int j = 0; j < layer->W->cols; ++j) {
                file << layer->W->getValues(i, j);
                if (j < layer->W->cols - 1) file << ",";
            }
            file << "\n";
        }

        // Save biases
        for (int i = 0; i < layer->b->rows; ++i) {
            file << layer->b->getValues(i);
            if (i < layer->b->rows - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}

void NeuralNetwork::load_config(std::string filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for reading: " << filepath << std::endl;
        return;
    }

    layers.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() == 3) {
            int input_num = std::stoi(tokens[0]);
            int output_num = std::stoi(tokens[1]);
            std::string activation = tokens[2];
            add_layer(new Layer(input_num, output_num, activation));
        }
    }

    file.close();
}

void NeuralNetwork::load_weights(std::string filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for reading: " << filepath << std::endl;
        return;
    }

    std::string line;
    for (Layer* layer : layers) {
        // Load W
        for (int i = 0; i < layer->W->rows; ++i) {
            if (std::getline(file, line)) {
                std::istringstream iss(line);
                std::string value;
                int j = 0;
                while (std::getline(iss, value, ',') && j < layer->W->cols) {
                    layer->W->setValue(i, j, std::stod(value));
                    j++;
                }
            }
        }

        // Load b
        if (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string value;
            int i = 0;
            while (std::getline(iss, value, ',') && i < layer->b->rows) {
                layer->b->setValue(i, std::stod(value));
                i++;
            }
        }
    }

    file.close();
}
