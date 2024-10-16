/**
 * @file neural_network_get_accuracy.cu
 * @brief Implementation of the NeuralNetwork::get_accuracy method.
 */
#include "neural_network.h"
#include <cuda_runtime.h>

__global__ void calculate_accuracy_kernel(const double* predictions, const double* Y, int size, int* correct_count) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within bounds
    if (idx < size) {
        // Increment correct_count if prediction matches true label
        if (predictions[idx] == Y[idx]) {
            atomicAdd(correct_count, 1);
        }
    }
}

double NeuralNetwork::get_accuracy(const Matrix& Y) const {
    // Get predictions
    Vector predictions = get_predictions();

    // Allocate device memory for correct count
    int* d_correct_count;
    cudaMalloc(&d_correct_count, sizeof(int));
    cudaMemset(d_correct_count, 0, sizeof(int));

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (Y.get_cols() + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to calculate accuracy
    calculate_accuracy_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        predictions.get_data(), Y.get_data(), Y.get_cols(), d_correct_count
    );

    // Copy correct count from device to host
    int h_correct_count;
    cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate accuracy
    double accuracy = static_cast<double>(h_correct_count) / Y.get_cols();

    // Free device memory
    cudaFree(d_correct_count);

    return accuracy;
}
