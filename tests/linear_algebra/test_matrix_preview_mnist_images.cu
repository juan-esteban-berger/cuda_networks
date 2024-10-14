/**
 * @file test_matrix_preview_mnist_images.cu
 * @brief Visual test for previewing MNIST images from X_train.csv
 */

#include <gtest/gtest.h>
#include "../src/linear_algebra/matrix.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

/**
 * @class MatrixPreviewMNISTImagesTest
 * @brief Test fixture for previewing MNIST images from X_train.csv
 */
class MatrixPreviewMNISTImagesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set the path to the X_train.csv file
        csv_path = "data/X_train.csv";
    }

    std::string csv_path;
};

/**
 * @test
 * @brief Read X_train.csv and preview a few random MNIST images
 */
TEST_F(MatrixPreviewMNISTImagesTest, PreviewRandomMNISTImages) {
    // Create a matrix to hold the entire X_train dataset
    Matrix X_train(60000, 784);  // MNIST has 60,000 training images, each 28x28 pixels

    // Read the CSV file into the matrix
    std::cout << "Reading X_train.csv..." << std::endl;
    X_train.read_csv(csv_path.c_str());
    std::cout << "X_train.csv loaded successfully." << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 59999);  // Generate random indices

    // Preview 5 random images
    for (int i = 0; i < 5; ++i) {
        int random_index = dis(gen);
        std::cout << "Previewing image at index " << random_index << ":" << std::endl;
        X_train.preview_image(random_index, 28, 28);
        std::cout << std::endl;
    }

    // Note: This test doesn't have any assertions.
    // It's a visual test to manually verify the images look correct.
    std::cout << "Please visually verify that the above images look like handwritten digits." << std::endl;
}
