#include <iostream>

#include "utils.h"

int main() {
////////////////////////////////////////////////////////////////////
    // Load Data
    std::cout << "Loading Data..." << std::endl;

    DataFrame X_train_original(60000, 784);
    X_train_original.read_csv_limited("data/X_train.csv", 0, 1000);

    DataFrame Y_train_original(60000, 10);
    Y_train_original.read_csv_limited("data/Y_train.csv", 0, 1000);

    std::cout << "X_Train: ";
    X_train_original.shape();
    std::cout << "Y_Train: ";
    Y_train_original.shape();

////////////////////////////////////////////////////////////////////
    // Tranpose Data
    DataFrame X_train = X_train_original.transpose();
    DataFrame Y_train = Y_train_original.transpose();

////////////////////////////////////////////////////////////////////
    // Normalize X Values
    X_train.normalize(0, 255);

    return 0;
}
