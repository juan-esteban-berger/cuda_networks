#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#include "cuda_linear_algebra.h"
#include "cuda_neural_network.h"

////////////////////////////////////////////////////////////////////////
// Main function
int main() {
////////////////////////////////////////////////////////////////////////
// Setup
    srand(time(NULL));
    printf("\n");

    dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
    int N = 64;
    dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

////////////////////////////////////////////////////////////////////////
// Define constants for number of rows and neurons
    const int NUM_ROWS_TRAIN = 1000;
    // const int NUM_ROWS_TEST = 10000;
    const int NUM_NEURONS_INPUT = 784;
    const int NUM_NEURONS_HIDDEN_1 = 200;
    const int NUM_NEURONS_HIDDEN_2 = 200;
    const int NUM_NEURONS_OUTPUT = 10;

////////////////////////////////////////////////////////////////////////
// Load, Preprocess, and Preview Data
    // Read in data from X_train.csv
    Matrix X_train;
    initialize_matrix(&X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);
    read_csv("data/X_train.csv", &X_train);
    printf("X_train:\n");
    preview_matrix(&X_train, 2);

    // Initialize Matrix d_X_train on device
    printf("Initializing Matrix d_X_train on device:\n");
    Matrix_GPU d_X_train;
    initialize_matrix_on_device(&d_X_train, NUM_ROWS_TRAIN, NUM_NEURONS_INPUT);

    // Copy Matrix X_train to device
    printf("Copying Matrix X_train to device:\n");
    copy_matrix_to_device(&X_train, &d_X_train);
    preview_matrix_GPU(&d_X_train, 2);

    // Read in data from Y_train.csv
    Matrix Y_train;
    initialize_matrix(&Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);
    read_csv("data/Y_train.csv", &Y_train);
    printf("Y_train:\n");
    preview_matrix(&Y_train, 2);

    // Initialize Matrix d_Y_train on device
    printf("Initializing Matrix d_Y_train on device:\n");
    Matrix_GPU d_Y_train;
    initialize_matrix_on_device(&d_Y_train, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);

    // Copy Matrix Y_train to device
    printf("Copying Matrix Y_train to device:\n");
    copy_matrix_to_device(&Y_train, &d_Y_train);
    preview_matrix_GPU(&d_Y_train, 2);

////////////////////////////////////////////////////////////////////////
// Test CUDA Vector Functions
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("Testing CUDA Vector Functions:\n");

    // Initialize Vector on host
    printf("Initializing Random Vector on host:\n");
    Vector v;
    initialize_vector(&v, 5);

    // Randomize Vector
    random_vector(&v);

    // Initialize Vector on device
    printf("Initializing Vector on device:\n");
    Vector d_v;
    initialize_vector_on_device(&d_v, 5);

    // Preview Original Vector
    printf("Previewing Original Vector:\n");
    preview_vector(&v, 2);

    // Copy Vector to device
    printf("Copying Vector to device:\n");
    copy_vector_to_device(&v, &d_v);

    // Free Vector
    printf("Freeing Vector on host:\n");
    free_vector(&v);

    // Initialize New Vector on host
    printf("Initializing New Vector on host:\n");
    Vector v2;
    initialize_vector(&v2, 5);

    // Copy Vector to host
    printf("Copying Vector to host:\n");
    copy_vector_to_host(&v2, &d_v);

    // Preview Vector Copied from device
    printf("Previewing Vector Copied from device:\n");
    preview_vector(&v2, 2);

    // Free New Vector
    printf("Freeing New Vector on host:\n");
    free_vector(&v2);

    // Free Vector on device
    printf("Freeing Vector on device:\n");
    free_vector_on_device(&d_v);

////////////////////////////////////////////////////////////////////////
// Test CUDA Matrix Functions
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("Testing CUDA Matrix Functions:\n");

    // Initialize Matrix on host
    printf("Initializing Random Matrix on host:\n");
    Matrix m;
    initialize_matrix(&m, 5, 5);

    // Randomize Matrix
    random_matrix(&m);

    // Initialize Matrix on device
    printf("Initializing Matrix on device:\n");
    Matrix_GPU d_m;
    initialize_matrix_on_device(&d_m, 5, 5);

    // Preview Original Matrix
    printf("Previewing Original Matrix:\n");
    preview_matrix(&m, 2);

    // Copy Matrix to device
    printf("Copying Matrix to device:\n");
    copy_matrix_to_device(&m, &d_m);

    // Free Matrix
    printf("Freeing Matrix on host:\n");
    free_matrix(&m);

    // Initialize New Matrix on host
    printf("Initializing New Matrix on host:\n");
    Matrix m2;
    initialize_matrix(&m2, 5, 5);

    // Copy Matrix to host
    printf("Copying Matrix to host:\n");
    copy_matrix_to_host(&m2, &d_m);

    // Preview Matrix Copied from device
    printf("Previewing Matrix Copied from device:\n");
    preview_matrix(&m2, 2);

    // Free New Matrix
    printf("Freeing New Matrix on host:\n");

    // Free Matrix on device
    printf("Freeing Matrix on device:\n");
    free_matrix_on_device(&d_m);

////////////////////////////////////////////////////////////////////////
// Test CPU Transpose Function
    // Initialize Matrix transpose_input on host
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("Testing CPU Transpose Function:\n");

    printf("Initializing Random Matrix transpose_input on host:\n");
    Matrix transpose_input;
    initialize_matrix(&transpose_input, 3, 2);
    random_matrix(&transpose_input);

    // Preview Matrix transpose_input
    printf("Previewing Matrix transpose_input:\n");
    preview_matrix(&transpose_input, 2);

    // Initialize Matrix transpose_output on host
    printf("Initializing Matrix transpose_output on host:\n");
    Matrix transpose_output;
    initialize_matrix(&transpose_output, 2, 3);

    // Transpose Matrix transpose_input
    printf("Transposing Matrix transpose_input:\n");
    transpose_matrix(&transpose_input, &transpose_output);


    // Preview Matrix transpose_output
    printf("Previewing Matrix transpose_output:\n");
    preview_matrix(&transpose_output, 2);

    // Free Matrix transpose_output
    printf("Freeing Matrix transpose_output on host:\n");
    free_matrix(&transpose_output);

////////////////////////////////////////////////////////////////////////
// Test CUDA Transpose Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("Testing CUDA Transpose Function:\n");

    // Initialize Matrix d_transpose_input on device
    printf("Initializing Random Matrix d_transpose_input on device:\n");
    Matrix_GPU d_transpose_input;
    initialize_matrix_on_device(&d_transpose_input, 3, 2);

    // Copy Matrix transpose_input to device
    printf("Copying Matrix transpose_input to device:\n");
    copy_matrix_to_device(&transpose_input, &d_transpose_input);

    // Initialize Matrix d_transpose_output on device
    printf("Initializing Matrix d_transpose_output on device:\n");
    Matrix_GPU d_transpose_output;
    initialize_matrix_on_device(&d_transpose_output, 2, 3);

    // Transpose Matrix d_transpose_input
    printf("Transposing Matrix d_transpose_input:\n");
    transpose_matrix_GPU <<< number_of_blocks, threads_per_block >>> (d_transpose_input.data,
		                                                     d_transpose_output.data,
								     d_transpose_input.rows,
								     d_transpose_input.cols);
    
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix transpose_output2 on host
    printf("Initializing Matrix transpose_output2 on host:\n");
    Matrix transpose_output2;
    initialize_matrix(&transpose_output2, 2, 3);

    // Copy Matrix transpose_output2 from device
    printf("Copying Matrix transpose_output2 from device:\n");
    copy_matrix_to_host(&transpose_output2, &d_transpose_output);

    // Preview Matrix transpose_output2
    printf("Previewing Matrix transpose_output2:\n");
    preview_matrix(&transpose_output2, 2);

    // Free Matrix transpose_input
    printf("Freeing Matrix transpose_input on host:\n");
    free_matrix(&transpose_input);

    // Free Matrix transpose_output2
    printf("Freeing Matrix transpose_output2 on host:\n");

    // Free Matrix d_transpose_input
    printf("Freeing Matrix d_transpose_input on device:\n");
    free_matrix_on_device(&d_transpose_input);

    // Free Matrix d_transpose_output
    printf("Freeing Matrix d_transpose_output on device:\n");
    free_matrix_on_device(&d_transpose_output);

////////////////////////////////////////////////////////////////////////
// Test CPU Matrix Multiplication
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("Testing CPU Matrix Multiplication:\n");

    // Initialize Matrix A on host
    printf("Initializing Random Matrix A on host:\n");
    Matrix A;
    initialize_matrix(&A, 2, 3);

    // Randomize Matrix A
    random_matrix(&A);
    preview_matrix(&A, 2);

    // Initialize Matrix B on host
    printf("Initializing Random Matrix B on host:\n");
    Matrix B;
    initialize_matrix(&B, 3, 2);

    // Randomize Matrix B
    random_matrix(&B);
    preview_matrix(&B, 2);

    // Multiply Matrix A by Matrix B
    printf("Multiplying Matrix A by Matrix B using CPU:\n");
    Matrix C;
    initialize_matrix(&C, 2, 2);
    matrix_multiply(&A, &B, &C);
    preview_matrix(&C, 2);


////////////////////////////////////////////////////////////////////////
// Test CUDA Matrix Multiplication
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("Testing CUDA Matrix Multiplication:\n");

    // Initialize Matrix d_A on device
    printf("Initializing Matrix d_A on device:\n");
    Matrix_GPU d_A;
    initialize_matrix_on_device(&d_A, 2, 3);

    // Copy Matrix A to device
    printf("Copying Matrix A to device:\n");
    copy_matrix_to_device(&A, &d_A);

    // Initialize Matrix d_B on device
    printf("Initializing Matrix d_B on device:\n");
    Matrix_GPU d_B;
    initialize_matrix_on_device(&d_B, 3, 2);

    // Copy Matrix B to device
    printf("Copying Matrix B to device:\n");
    copy_matrix_to_device(&B, &d_B);

    // Initialize Matrix d_C on device
    printf("Initializing Matrix d_C on device:\n");
    Matrix_GPU d_C;
    initialize_matrix_on_device(&d_C, 2, 2);

    // Free Matrix A
    printf("Freeing Matrix A on host:\n");
    free_matrix(&A);

    // Free Matrix B
    printf("Freeing Matrix B on host:\n");
    free_matrix(&B);

    // Multiply using: __global__ void matrix_multiply_GPU(float *a, float *b, float *c, int aRows, int aCols, int bCols);
    printf("Multiplying Matrix d_A by Matrix d_B using GPU:\n");
    matrix_multiply_GPU <<< number_of_blocks, threads_per_block >>> (d_A.data,
		                                                     d_B.data,
								     d_C.data,
								     d_A.rows,
								     d_A.cols,
								     d_B.cols);
    
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
    
    // Initialize Matrix C2 on host
    printf("Initializing Matrix C2 on host:\n");
    Matrix C2;
    initialize_matrix(&C2, 2, 2);

    // Copy Matrix C2 from device
    printf("Copying Matrix C from device:\n");
    copy_matrix_to_host(&C2, &d_C);

    // Preview Matrix C2
    printf("Previewing Matrix C2:\n");
    preview_matrix(&C2, 2);

    // Free Matrix d_A
    printf("Freeing Matrix d_A on device:\n");
    free_matrix_on_device(&d_A);

    // Free Matrix d_B
    printf("Freeing Matrix d_B on device:\n");
    free_matrix_on_device(&d_B);

    // Free Matrix C2
    printf("Freeing Matrix C on host:\n");
    free_matrix(&C2);

    // Free Matrix d_C
    printf("Freeing Matrix d_C on device:\n");
    free_matrix_on_device(&d_C);

////////////////////////////////////////////////////////////////////////
// Test CPU Elementwise Matrix Multiplication
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("Testing CPU Elementwise Matrix Multiplication:\n");

    // Initialize Matrix A_em on host
    printf("Initializing Random Matrix A_em on host:\n");
    Matrix A_em;
    initialize_matrix(&A_em, 2, 3);

    // Randomize Matrix A_em
    random_matrix(&A_em);
    preview_matrix(&A_em, 2);

    // Initialize Matrix B_em on host
    printf("Initializing Random Matrix B_em on host:\n");
    Matrix B_em;
    initialize_matrix(&B_em, 2, 3);

    // Randomize Matrix B_em
    random_matrix(&B_em);
    preview_matrix(&B_em, 2);

    // Multiply Matrix A_em by Matrix B_em
    printf("Elementwise Multiplying Matrix A_em by Matrix B_em using CPU:\n");
    Matrix C_em;
    initialize_matrix(&C_em, 2, 3);
    matrix_multiply_elementwise(&A_em, &B_em, &C_em);

    // Preview Matrix C_em
    preview_matrix(&C_em, 2);

////////////////////////////////////////////////////////////////////////
// Test CUDA Elementwise Matrix Multiplication
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("Testing CUDA Elementwise Matrix Multiplication:\n");

    // Initialize Matrix d_A_em on device
    printf("Initializing Matrix d_A_em on device:\n");
    Matrix_GPU d_A_em;
    initialize_matrix_on_device(&d_A_em, 2, 3);

    // Copy Matrix A_em to device
    printf("Copying Matrix A_em to device:\n");
    copy_matrix_to_device(&A_em, &d_A_em);

    // Initialize Matrix d_B_em on device
    printf("Initializing Matrix d_B_em on device:\n");
    Matrix_GPU d_B_em;
    initialize_matrix_on_device(&d_B_em, 2, 3);

    // Copy Matrix B_em to device
    printf("Copying Matrix B_em to device:\n");
    copy_matrix_to_device(&B_em, &d_B_em);

    // Initialize Matrix d_C_em on device
    printf("Initializing Matrix d_C_em on device:\n");
    Matrix_GPU d_C_em;
    initialize_matrix_on_device(&d_C_em, 2, 3);

    // Initialize Matrix C_em2 on host
    printf("Initializing Matrix C_em2 on host:\n");
    Matrix C_em2;
    initialize_matrix(&C_em2, 2, 3);

    // Multiply using: __global__ void matrix_multiply_elementwise_GPU(float *a, float *b, float *c, int aRows, int aCols);
    printf("Elementwise Multiplying Matrix d_A_em by Matrix d_B_em using GPU:\n");
    matrix_multiply_elementwise_GPU <<< number_of_blocks, threads_per_block >>> (d_A_em.data,
		                                                                 d_B_em.data,
								                 d_C_em.data,
								                 d_A_em.rows,
								                 d_A_em.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Copy Matrix C_em2 from device
    printf("Copying Matrix C_em2 from device:\n");
    copy_matrix_to_host(&C_em2, &d_C_em);

    // Preview Matrix C_em2
    printf("Previewing Matrix C_em2:\n");
    preview_matrix(&C_em2, 2);

    // Free Matrix A_em
    printf("Freeing Matrix A_em on host:\n");
    free_matrix(&A_em);

    // Free Matrix B_em
    printf("Freeing Matrix B_em on host:\n");
    free_matrix(&B_em);

    // Free Matrix C_em
    printf("Freeing Matrix C_em on host:\n");
    free_matrix(&C_em);

    // Free Matrix C_em2
    printf("Freeing Matrix C_em2 on host:\n");
    free_matrix(&C_em2);

    // Free Matrix d_A_em
    printf("Freeing Matrix d_A_em on device:\n");
    free_matrix_on_device(&d_A_em);

    // Free Matrix d_B_em
    printf("Freeing Matrix d_B_em on device:\n");
    free_matrix_on_device(&d_B_em);

    // Free Matrix d_C_em
    printf("Freeing Matrix d_C_em on device:\n");
    free_matrix_on_device(&d_C_em);

////////////////////////////////////////////////////////////////////////
// Test CPU Elementwise Matrix Subtraction
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CPU Elementwise Matrix Subtraction:\n");

    // Initialize Matrix A_es on host
    printf("Initializing Random Matrix A_es on host:\n");
    Matrix A_es;
    initialize_matrix(&A_es, 2, 3);

    // Randomize Matrix A_es
    random_matrix(&A_es);
    preview_matrix(&A_es, 2);

    // Initialize Matrix B_es on host
    printf("Initializing Random Matrix B_es on host:\n");
    Matrix B_es;
    initialize_matrix(&B_es, 2, 3);

    // Randomize Matrix B_es
    random_matrix(&B_es);
    preview_matrix(&B_es, 2);

    // Subtract Matrix A_es by Matrix B_es
    printf("Elementwise Subtracting Matrix A_es by Matrix B_es using CPU:\n");
    Matrix C_es;
    initialize_matrix(&C_es, 2, 3);
    matrix_subtract(&A_es, &B_es, &C_es);

    // Preview Matrix C_es
    preview_matrix(&C_es, 2);

////////////////////////////////////////////////////////////////////////
// Test CUDA Elementwise Matrix Subtraction
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CUDA Elementwise Matrix Subtraction:\n");

    // Initialize Matrix d_A_es on device
    printf("Initializing Matrix d_A_es on device:\n");
    Matrix_GPU d_A_es;
    initialize_matrix_on_device(&d_A_es, 2, 3);

    // Copy Matrix A_es to device
    printf("Copying Matrix A_es to device:\n");
    copy_matrix_to_device(&A_es, &d_A_es);

    // Initialize Matrix d_B_es on device
    printf("Initializing Matrix d_B_es on device:\n");
    Matrix_GPU d_B_es;
    initialize_matrix_on_device(&d_B_es, 2, 3);

    // Copy Matrix B_es to device
    printf("Copying Matrix B_es to device:\n");
    copy_matrix_to_device(&B_es, &d_B_es);

    // Initialize Matrix d_C_es on device
    printf("Initializing Matrix d_C_es on device:\n");
    Matrix_GPU d_C_es;
    initialize_matrix_on_device(&d_C_es, 2, 3);

    // Perform Elementwise Subtraction using: __global__ void matrix_subtract_GPU(float *a, float *b, float *c, int aRows, int aCols);
    printf("Elementwise Subtracting Matrix d_A_es by Matrix d_B_es using GPU:\n");
    matrix_subtract_GPU <<< number_of_blocks, threads_per_block >>> (d_A_es.data,
		                                                   d_B_es.data,
								   d_C_es.data,
								   d_A_es.rows,
								   d_A_es.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix C_es2 on host
    printf("Initializing Matrix C_es2 on host:\n");
    Matrix C_es2;
    initialize_matrix(&C_es2, 2, 3);

    // Copy Matrix C_es2 from device
    printf("Copying Matrix C_es2 from device:\n");
    copy_matrix_to_host(&C_es2, &d_C_es);

    // Preview Matrix C_es2
    printf("Previewing Matrix C_es2:\n");
    preview_matrix(&C_es2, 2);

    // Free Matrix A_es
    printf("Freeing Matrix A_es on host:\n");
    free_matrix(&A_es);

    // Free Matrix B_es
    printf("Freeing Matrix B_es on host:\n");
    free_matrix(&B_es);

    // Free Matrix C_es
    printf("Freeing Matrix C_es on host:\n");
    free_matrix(&C_es);

    // Free Matrix C_es2
    printf("Freeing Matrix C_es2 on host:\n");
    free_matrix(&C_es2);

    // Free Matrix d_A_es
    printf("Freeing Matrix d_A_es on device:\n");
    free_matrix_on_device(&d_A_es);

    // Free Matrix d_B_es
    printf("Freeing Matrix d_B_es on device:\n");
    free_matrix_on_device(&d_B_es);

    // Free Matrix d_C_es
    printf("Freeing Matrix d_C_es on device:\n");
    free_matrix_on_device(&d_C_es);

////////////////////////////////////////////////////////////////////////
// Test CPU Divide Matrix by Scalar
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CPU Divide Matrix by Scalar:\n");

    // Initialize Matrix A_ds on host
    printf("Initializing Random Matrix A_ds on host:\n");
    Matrix A_ds;
    initialize_matrix(&A_ds, 2, 3);

    // Randomize Matrix A_ds
    random_matrix(&A_ds);
    preview_matrix(&A_ds, 2);

    // Define Scalar
    float scalar = 4.0;

    // Divide Matrix A_ds by Scalar
    printf("Dividing Matrix A_ds by Scalar using CPU:\n");
    divide_matrix_by_scalar(&A_ds, scalar);

    // Preview Matrix A_ds
    preview_matrix(&A_ds, 2);

    // Free Matrix A_ds
    printf("Freeing Matrix A_ds on host:\n");
    free_matrix(&A_ds);

////////////////////////////////////////////////////////////////////////
// Test CUDA Divide Matrix by Scalar
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CUDA Divide Matrix by Scalar:\n");

    // Initialize a second Matrix A_ds2 on host
    printf("Initializing Random Matrix A_ds2 on host:\n");
    Matrix A_ds2;
    initialize_matrix(&A_ds2, 2, 3);

    // Randomize Matrix A_ds2
    random_matrix(&A_ds2);
    preview_matrix(&A_ds2, 2);

    // Define Scalar
    float scalar2 = 4.0;

    // Initialize Matrix d_A_ds on device
    printf("Initializing Matrix d_A_ds on device:\n");
    Matrix_GPU d_A_ds;
    initialize_matrix_on_device(&d_A_ds, 2, 3);

    // Copy Matrix A_ds2 to device
    printf("Copying Matrix A_ds2 to device:\n");
    copy_matrix_to_device(&A_ds2, &d_A_ds);

    // Divide Matrix d_A_ds by Scalar
    printf("Dividing Matrix d_A_ds by Scalar using GPU:\n");
    divide_matrix_by_scalar_GPU <<< number_of_blocks, threads_per_block >>> (d_A_ds.data,
		                                                             scalar2,
								             d_A_ds.rows,
								             d_A_ds.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix A_ds3 on host
    printf("Initializing Matrix A_ds3 on host:\n");
    Matrix A_ds3;
    initialize_matrix(&A_ds3, 2, 3);

    // Copy Matrix A_ds3 from device
    printf("Copying Matrix A_ds3 from device:\n");
    copy_matrix_to_host(&A_ds3, &d_A_ds);

    // Preview Matrix A_ds3
    preview_matrix(&A_ds3, 2);

    // Free Matrix A_ds2
    printf("Freeing Matrix A_ds2 on host:\n");
    free_matrix(&A_ds2);

    // Free Matrix A_ds3
    printf("Freeing Matrix A_ds3 on host:\n");
    free_matrix(&A_ds3);

    // Free Matrix d_A_ds
    printf("Freeing Matrix d_A_ds on device:\n");
    free_matrix_on_device(&d_A_ds);

///////////////////////////////////////////////////////////////////////
// Test Sum all elements in Matrix CPU Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing Sum all elements in Matrix CPU Function:\n");

    // Initialize Matrix A_sum on host
    printf("Initializing Random Matrix A_sum on host:\n");
    Matrix A_sum;
    initialize_matrix(&A_sum, 2, 3);

    // Randomize Matrix A_sum
    random_matrix(&A_sum);
    preview_matrix(&A_sum, 2);

    // Sum all elements in Matrix A_sum
    printf("Summing all elements in Matrix A_sum using CPU:\n");
    float sum;
    sum_matrix(&A_sum, &sum);

    // Preview Sum
    printf("Sum: %f\n", sum);

////////////////////////////////////////////////////////////////////////
// Test Sum all elements in Matrix CUDA Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing Sum all elements in Matrix CUDA Function:\n");
    
    // Initialize Matrix d_A_sum on device
    printf("Initializing Matrix d_A_sum on device:\n");
    Matrix_GPU d_A_sum;
    initialize_matrix_on_device(&d_A_sum, 2, 3);
    
    // Copy Matrix A_sum to device
    printf("Copying Matrix A_sum to device:\n");
    copy_matrix_to_device(&A_sum, &d_A_sum);
    
    // Allocate memory for the result on the GPU
    printf("Allocating memory for the result on the GPU:\n");
    float *d_sum;
    cudaMalloc((void **)&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float)); // Initialize the sum to 0
    
    // Calculate the number of threads and blocks
    int threadsPerBlock = 256; // This should be a power of two
    int blocks = ((d_A_sum.rows * d_A_sum.cols) + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel with the correct amount of shared memory
    printf("Launching the kernel to sum all elements in Matrix d_A_sum using GPU:\n");
    sum_matrix_GPU<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A_sum.data,
		                                                                 d_sum,
										 d_A_sum.rows,
										 d_A_sum.cols);
    
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
    
    // Copy the result back to the host
    float h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Preview Sum
    printf("Sum: %f\n", h_sum);
    
    // Free the GPU memory
    cudaFree(d_sum);
    
    // Free Matrix A_sum
    printf("Freeing Matrix A_sum on host:\n");
    free_matrix(&A_sum);
    
    // Free Matrix d_A_sum
    printf("Freeing Matrix d_A_sum on device:\n");
    free_matrix_on_device(&d_A_sum);

////////////////////////////////////////////////////////////////////////
// Test CPU Add Vector to Matrix
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CPU Add Vector to Matrix:\n");

    // Initialize Matrix A_avm on host
    printf("Initializing Random Matrix A_avm on host:\n");
    Matrix A_avm;
    initialize_matrix(&A_avm, 2, 3);

    // Randomize Matrix A_avm
    random_matrix(&A_avm);

    // Preview Matrix A_avm
    preview_matrix(&A_avm, 2);

    // Initialize Vector v_avm on host
    printf("Initializing Random Vector v_avm on host:\n");
    Vector v_avm;
    initialize_vector(&v_avm, 3);

    // Randomize Vector v_avm
    random_vector(&v_avm);

    // Preview Vector v_avm
    preview_vector(&v_avm, 2);

    // Add Vector v_avm to Matrix A_avm
    printf("Adding Vector v_avm to Matrix A_avm using CPU:\n");
    add_vector_to_matrix(&A_avm, &v_avm);

    // Preview Matrix A_avm
    preview_matrix(&A_avm, 2);

////////////////////////////////////////////////////////////////////////
// Test CUDA Add Vector to Matrix
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    dim3 threads_per_block2 (16, 16, 1); // A 16 x 16 block threads
    int N2 = 32;
    dim3 number_of_blocks2 ((N2 / threads_per_block2.x) + 1, (N2 / threads_per_block2.y) + 1, 1);

    printf("Testing CUDA Add Vector to Matrix:\n");

    // Initialize Matrix d_A_avm on device
    printf("Initializing Matrix d_A_avm on device:\n");
    Matrix_GPU d_A_avm;
    initialize_matrix_on_device(&d_A_avm, 2, 3);

    // Copy Matrix A_avm to device
    printf("Copying Matrix A_avm to device:\n");
    copy_matrix_to_device(&A_avm, &d_A_avm);

    // Initialize Vector d_v_avm on device
    printf("Initializing Vector d_v_avm on device:\n");
    Vector d_v_avm;
    initialize_vector_on_device(&d_v_avm, 3);

    // Copy Vector v_avm to device
    printf("Copying Vector v_avm to device:\n");
    copy_vector_to_device(&v_avm, &d_v_avm);

    // Add Vector d_v_avm to Matrix d_A_avm
    printf("Adding Vector d_v_avm to Matrix d_A_avm using GPU:\n");
    add_vector_to_matrix_GPU <<< number_of_blocks2, threads_per_block2 >>> (d_A_avm.data,
		                                                            d_v_avm.data,
								            d_A_avm.rows,
								            d_A_avm.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix A_avm2 on host
    printf("Initializing Matrix A_avm2 on host:\n");
    Matrix A_avm2;
    initialize_matrix(&A_avm2, 2, 3);

    // Copy Matrix A_avm2 from device
    printf("Copying Matrix A_avm2 from device:\n");
    copy_matrix_to_host(&A_avm2, &d_A_avm);

    // Preview Matrix A_avm2
    preview_matrix(&A_avm2, 2);

    // Free Matrix A_avm
    printf("Freeing Matrix A_avm on host:\n");
    free_matrix(&A_avm);

    // Free Matrix A_avm2
    printf("Freeing Matrix A_avm2 on host:\n");
    free_matrix(&A_avm2);

    // Free Vector v_avm
    printf("Freeing Vector v_avm on host:\n");
    free_vector(&v_avm);

    // Free Matrix d_A_avm
    printf("Freeing Matrix d_A_avm on device:\n");
    free_matrix_on_device(&d_A_avm);

    // Free Vector d_v_avm
    printf("Freeing Vector d_v_avm on device:\n");
    free_vector_on_device(&d_v_avm);

////////////////////////////////////////////////////////////////////////
// Test CPU Argmax Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CPU Argmax Function:\n");

    // Initialize Matrix A_argmax on host
    printf("Initializing Random Matrix A_argmax on host:\n");
    Matrix A_argmax;
    initialize_matrix(&A_argmax, 2, 3);

    // Randomize Matrix A_argmax
    random_matrix(&A_argmax);

    // Preview Matrix A_argmax
    preview_matrix(&A_argmax, 2);

    // Initialize Vector v_argmax on host
    printf("Initializing Vector v_argmax on host:\n");
    Vector v_argmax;
    initialize_vector(&v_argmax, 2);

    // Argmax Matrix A_argmax
    argmax(&A_argmax, &v_argmax);

    // Preview Vector v_argmax
    preview_vector(&v_argmax, 2);

    // Free Vector v_argmax
    printf("Freeing Vector v_argmax on host:\n");
    free_vector(&v_argmax);

////////////////////////////////////////////////////////////////////////
// Test CUDA Argmax Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    
    printf("Testing CUDA Argmax Function:\n");
    
    // Initialize Matrix d_A_argmax on device
    printf("Initializing Matrix d_A_argmax on device:\n");
    Matrix_GPU d_A_argmax;
    initialize_matrix_on_device(&d_A_argmax, 2, 3);
    
    // Copy Matrix A_argmax to device
    printf("Copying Matrix A_argmax to device:\n");
    copy_matrix_to_device(&A_argmax, &d_A_argmax);
    
    // Initialize Vector d_v_argmax on device
    printf("Initializing Vector d_v_argmax on device:\n");
    Vector d_v_argmax;
    initialize_vector_on_device(&d_v_argmax, 2); // The length should match the number of columns in A_argmax
    
    // Calculate the number of blocks needed for the argmax operation
    int blocksPerGrid = (d_A_argmax.cols + threads_per_block.x - 1) / threads_per_block.x;
    
    // Argmax Matrix d_A_argmax
    argmax_GPU <<< blocksPerGrid, threads_per_block.x >>> (d_A_argmax.data,
                                                           d_v_argmax.data,
                                                           d_A_argmax.rows,
                                                           d_A_argmax.cols);
    
    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
    
    // Initialize Vector v_argmax2 on host
    printf("Initializing Vector v_argmax2 on host:\n");
    Vector v_argmax2;
    initialize_vector(&v_argmax2, 2); // The length should match the number of columns in A_argmax
    
    // Copy Vector v_argmax2 from device
    printf("Copying Vector v_argmax2 from device:\n");
    copy_vector_to_host(&v_argmax2, &d_v_argmax);
    
    // Preview Vector v_argmax2
    preview_vector(&v_argmax2, 2);

////////////////////////////////////////////////////////////////////////
// Test ReLU Activation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing ReLU Activation Function:\n");

    // Initialize Matrix A_relu on host
    printf("Initializing Random Matrix A_relu on host:\n");
    Matrix A_relu;
    initialize_matrix(&A_relu, 2, 3);

    // Randomize Matrix A_relu
    random_matrix(&A_relu);

    // Preview Matrix A_relu
    preview_matrix(&A_relu, 2);

    // Initialize Matrix B_relu on host
    printf("Initializing Matrix B_relu on host:\n");
    Matrix B_relu;
    initialize_matrix(&B_relu, 2, 3);

    // ReLU Matrix A_relu
    ReLU(&A_relu, &B_relu);

    // Preview Matrix B_relu
    preview_matrix(&B_relu, 2);

////////////////////////////////////////////////////////////////////////
// Test CUDA ReLU Activation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CUDA ReLU Activation Function:\n");

    // Initialize Matrix d_A_relu on device
    printf("Initializing Matrix d_A_relu on device:\n");
    Matrix_GPU d_A_relu;
    initialize_matrix_on_device(&d_A_relu, 2, 3);

    // Copy Matrix A_relu to device
    printf("Copying Matrix A_relu to device:\n");
    copy_matrix_to_device(&A_relu, &d_A_relu);

    // Initialize Matrix d_B_relu on device
    printf("Initializing Matrix d_B_relu on device:\n");
    Matrix_GPU d_B_relu;
    initialize_matrix_on_device(&d_B_relu, 2, 3);

    // ReLU Matrix d_A_relu
    ReLU_GPU <<< number_of_blocks, threads_per_block >>> (d_A_relu.data,
		                                        d_B_relu.data,
							d_A_relu.rows,
							d_A_relu.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix B_relu2 on host
    printf("Initializing Matrix B_relu2 on host:\n");
    Matrix B_relu2;
    initialize_matrix(&B_relu2, 2, 3);

    // Copy Matrix B_relu2 from device
    printf("Copying Matrix B_relu2 from device:\n");
    copy_matrix_to_host(&B_relu2, &d_B_relu);

    // Preview Matrix B_relu2
    preview_matrix(&B_relu2, 2);

    // Free Matrix A_relu
    printf("Freeing Matrix A_relu on host:\n");
    free_matrix(&A_relu);

    // Free Matrix B_relu
    printf("Freeing Matrix B_relu on host:\n");
    free_matrix(&B_relu);

    // Free Matrix B_relu2
    printf("Freeing Matrix B_relu2 on host:\n");
    free_matrix(&B_relu2);

    // Free Matrix d_A_relu
    printf("Freeing Matrix d_A_relu on device:\n");
    free_matrix_on_device(&d_A_relu);

    // Free Matrix d_B_relu
    printf("Freeing Matrix d_B_relu on device:\n");
    free_matrix_on_device(&d_B_relu);

////////////////////////////////////////////////////////////////////////
// Test ReLU Derivative Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing ReLU Derivative Function:\n");

    // Initialize Matrix A_relu_d on host
    printf("Initializing Random Matrix A_relu_d on host:\n");
    Matrix A_relu_d;
    initialize_matrix(&A_relu_d, 2, 3);

    // Randomize Matrix A_relu_d
    random_matrix(&A_relu_d);

    // Preview Matrix A_relu_d
    preview_matrix(&A_relu_d, 2);

    // Initialize Matrix B_relu_d on host
    printf("Initializing Matrix B_relu_d on host:\n");
    Matrix B_relu_d;
    initialize_matrix(&B_relu_d, 2, 3);

    // ReLU Derivative Matrix A_relu_d
    ReLU_derivative(&A_relu_d, &B_relu_d);

    // Preview Matrix B_relu_d
    preview_matrix(&B_relu_d, 2);

////////////////////////////////////////////////////////////////////////
// Test CUDA ReLU Derivative Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CUDA ReLU Derivative Function:\n");

    // Initialize Matrix d_A_relu_d on device
    printf("Initializing Matrix d_A_relu_d on device:\n");
    Matrix_GPU d_A_relu_d;
    initialize_matrix_on_device(&d_A_relu_d, 2, 3);

    // Copy Matrix A_relu_d to device
    printf("Copying Matrix A_relu_d to device:\n");
    copy_matrix_to_device(&A_relu_d, &d_A_relu_d);

    // Initialize Matrix d_B_relu_d on device
    printf("Initializing Matrix d_B_relu_d on device:\n");
    Matrix_GPU d_B_relu_d;
    initialize_matrix_on_device(&d_B_relu_d, 2, 3);

    // ReLU Derivative Matrix d_A_relu_d
    ReLU_derivative_GPU <<< number_of_blocks, threads_per_block >>> (d_A_relu_d.data,
		                                                d_B_relu_d.data,
								d_A_relu_d.rows,
								d_A_relu_d.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix B_relu_d2 on host
    printf("Initializing Matrix B_relu_d2 on host:\n");
    Matrix B_relu_d2;
    initialize_matrix(&B_relu_d2, 2, 3);

    // Copy Matrix B_relu_d2 from device
    printf("Copying Matrix B_relu_d2 from device:\n");
    copy_matrix_to_host(&B_relu_d2, &d_B_relu_d);

    // Preview Matrix B_relu_d2
    preview_matrix(&B_relu_d2, 2);

    // Free Matrix A_relu_d
    printf("Freeing Matrix A_relu_d on host:\n");
    free_matrix(&A_relu_d);

    // Free Matrix B_relu_d
    printf("Freeing Matrix B_relu_d on host:\n");
    free_matrix(&B_relu_d);

    // Free Matrix B_relu_d2
    printf("Freeing Matrix B_relu_d2 on host:\n");
    free_matrix(&B_relu_d2);

    // Free Matrix d_A_relu_d
    printf("Freeing Matrix d_A_relu_d on device:\n");
    free_matrix_on_device(&d_A_relu_d);

    // Free Matrix d_B_relu_d
    printf("Freeing Matrix d_B_relu_d on device:\n");
    free_matrix_on_device(&d_B_relu_d);

////////////////////////////////////////////////////////////////////////
// Test Softmax Activation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing Softmax Activation Function:\n");

    // Initialize Matrix A_softmax on host
    printf("Initializing Random Matrix A_softmax on host:\n");
    Matrix A_softmax;
    initialize_matrix(&A_softmax, 2, 3);

    // Randomize Matrix A_softmax
    random_matrix(&A_softmax);

    // Preview Matrix A_softmax
    preview_matrix(&A_softmax, 2);

    // Initialize Matrix B_softmax on host
    printf("Initializing Matrix B_softmax on host:\n");
    Matrix B_softmax;
    initialize_matrix(&B_softmax, 2, 3);

    // Softmax Matrix A_softmax
    softmax(&A_softmax, &B_softmax);

    // Preview Matrix B_softmax
    preview_matrix(&B_softmax, 2);

////////////////////////////////////////////////////////////////////////
// Test CUDA Softmax Activation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing CUDA Softmax Activation Function:\n");

    // Initialize Matrix d_A_softmax on device
    printf("Initializing Matrix d_A_softmax on device:\n");
    Matrix_GPU d_A_softmax;
    initialize_matrix_on_device(&d_A_softmax, 2, 3);

    // Copy Matrix A_softmax to device
    printf("Copying Matrix A_softmax to device:\n");
    copy_matrix_to_device(&A_softmax, &d_A_softmax);

    // Initialize Matrix d_B_softmax on device
    printf("Initializing Matrix d_B_softmax on device:\n");
    Matrix_GPU d_B_softmax;
    initialize_matrix_on_device(&d_B_softmax, 2, 3);

    // Softmax Matrix d_A_softmax
    int cols = d_A_softmax.cols;
    int sharedMemSize = cols * sizeof(float) * threadsPerBlock;
    softmax_GPU <<< number_of_blocks, threads_per_block, sharedMemSize >>> (d_A_softmax.data,
		                                          d_B_softmax.data,
							  d_A_softmax.rows,
							  d_A_softmax.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix B_softmax2 on host
    printf("Initializing Matrix B_softmax2 on host:\n");
    Matrix B_softmax2;
    initialize_matrix(&B_softmax2, 2, 3);

    // Copy Matrix B_softmax2 from device
    printf("Copying Matrix B_softmax2 from device:\n");
    copy_matrix_to_host(&B_softmax2, &d_B_softmax);

    // Preview Matrix B_softmax2
    preview_matrix(&B_softmax2, 2);

    // Free Matrix A_softmax
    printf("Freeing Matrix A_softmax on host:\n");
    free_matrix(&A_softmax);

    // Free Matrix B_softmax
    printf("Freeing Matrix B_softmax on host:\n");
    free_matrix(&B_softmax);

    // Free Matrix B_softmax2
    printf("Freeing Matrix B_softmax2 on host:\n");
    free_matrix(&B_softmax2);

    // Free Matrix d_A_softmax
    printf("Freeing Matrix d_A_softmax on device:\n");
    free_matrix_on_device(&d_A_softmax);

    // Free Matrix d_B_softmax
    printf("Freeing Matrix d_B_softmax on device:\n");
    free_matrix_on_device(&d_B_softmax);

////////////////////////////////////////////////////////////////////////
// Test CPU Calculate Accuracy Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    
    printf("Testing CPU Calculate Accuracy Function:\n");

    // Initialize Vector A_acc on host
    printf("Initializing Random Vector A_acc on host:\n");
    Vector A_acc;
    initialize_vector(&A_acc, 3);

    // Set values of Vector A_acc to 1, 2, 3
    A_acc.data[0] = 1;
    A_acc.data[1] = 2;
    A_acc.data[2] = 3;

    // Preview Vector A_acc
    preview_vector(&A_acc, 2);

    // Initialize Vector B_acc on host
    printf("Initializing Vector B_acc on host:\n");
    Vector B_acc;
    initialize_vector(&B_acc, 3);

    // Set values of Vector B_acc to 1, 2, 4
    B_acc.data[0] = 1;
    B_acc.data[1] = 2;
    B_acc.data[2] = 4;

    // Preview Vector B_acc
    preview_vector(&B_acc, 2);

    // Calculate Accuracy of Vector A_acc
    printf("Calculating Accuracy of Vector A_acc using CPU:\n");
    calculate_accuracy(&A_acc, &B_acc);

////////////////////////////////////////////////////////////////////////
// Test CUDA Calculate Accuracy Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    // Randomize Vector A_acc
    preview_vector(&A_acc, 2);

    // Randomize Vector B_acc
    preview_vector(&B_acc, 2);

    // Initialize Vector d_A_acc on device
    printf("Initializing Vector d_A_acc on device:\n");
    Vector d_A_acc;
    initialize_vector_on_device(&d_A_acc, 3);

    // Copy Vector A_acc to device
    printf("Copying Vector A_acc to device:\n");
    copy_vector_to_device(&A_acc, &d_A_acc);

    // Preview Vector d_A_acc
    printf("Previewing Vector d_A_acc:\n");
    preview_vector_GPU(&d_A_acc, 2);

    // Initialize Vector d_B_acc on device
    printf("Initializing Vector d_B_acc on device:\n");
    Vector d_B_acc;
    initialize_vector_on_device(&d_B_acc, 3);

    // Copy Vector B_acc to device
    printf("Copying Vector B_acc to device:\n");
    copy_vector_to_device(&B_acc, &d_B_acc);

    // Preview Vector d_B_acc
    printf("Previewing Vector d_B_acc:\n");
    preview_vector_GPU(&d_B_acc, 2);

    // Initialize float accuracy on device
    printf("Initializing float accuracy on device:\n");
    float *d_accuracy;
    cudaMalloc((void **)&d_accuracy, sizeof(float));

    // Calculate Accuracy for Vectors d_A_acc and d_B_acc
    printf("Calculating Accuracy of Vector d_A_acc using GPU:\n");
    calculate_accuracy_GPU <<< number_of_blocks, threads_per_block, sharedMemSize >>> (d_A_acc.data,
		                                                                       d_B_acc.data,
					                                               d_A_acc.rows,
					                                               d_accuracy);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize float accuracy on host
    printf("Initializing float accuracy on host:\n");
    float accuracy;

    // Copy float accuracy from device
    printf("Copying float accuracy from device:\n");
    cudaMemcpy(&accuracy, d_accuracy, sizeof(float), cudaMemcpyDeviceToHost);

    // Preview float accuracy
    printf("Accuracy: %f\n", accuracy);

    // Free Vector A_acc
    printf("Freeing Vector A_acc on host:\n");
    free_vector(&A_acc);

    // Free Vector B_acc
    printf("Freeing Vector B_acc on host:\n");
    free_vector(&B_acc);

    // Free Vector d_A_acc
    printf("Freeing Vector d_A_acc on device:\n");
    free_vector_on_device(&d_A_acc);

    // Free Vector d_B_acc
    printf("Freeing Vector d_B_acc on device:\n");
    free_vector_on_device(&d_B_acc);

    // Free float d_accuracy
    printf("Freeing float d_accuracy on device:\n");
    cudaFree(d_accuracy);
    
////////////////////////////////////////////////////////////////////////
// Test CPU Forward Propagation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    // Initialize Neural Network
    printf("Initializing Neural Network:\n");
    NeuralNetwork nn_value;
    initialize_neural_network(&nn_value,
		    	      NUM_NEURONS_INPUT,
			      NUM_NEURONS_HIDDEN_1,
			      NUM_NEURONS_HIDDEN_2,
			      NUM_NEURONS_OUTPUT);

    // Preview Input Layer Weights
    printf("Previewing Input Layer Weights:\n");
    preview_matrix(&(nn_value.W1), 2);

    // Preview Input Layer Biases
    printf("Previewing Input Layer Biases:\n");
    preview_vector(&(nn_value.b1), 2);

    // Preview Hidden Layer 1 Weights
    printf("Previewing Hidden Layer 1 Weights:\n");
    preview_matrix(&(nn_value.W2), 2);

    // Preview Hidden Layer 1 Biases
    printf("Previewing Hidden Layer 1 Biases:\n");
    preview_vector(&(nn_value.b2), 2);

    // Preview Hidden Layer 2 Weights
    printf("Previewing Hidden Layer 2 Weights:\n");
    preview_matrix(&(nn_value.WOutput), 2);

    // Preview Hidden Layer 2 Biases
    printf("Previewing Hidden Layer 2 Biases:\n");
    preview_vector(&(nn_value.bOutput), 2);

    // Create pointer to Neural Network
    NeuralNetwork *nn = &nn_value;

    // Transpose X to get the correct dimensions for matrix multiplication
    Matrix X_T;
    initialize_matrix(&X_T, X_train.cols, X_train.rows);
    transpose_matrix(&X_train, &X_T);

    // Transpose Y_T to match AOutput
    Matrix Y_T;
    initialize_matrix(&Y_T, Y_train.cols, Y_train.rows);
    transpose_matrix(&Y_train, &Y_T);

    // First Layer
    Matrix Z1;
    initialize_matrix(&Z1, nn->W1.rows, X_T.cols);
    Matrix A1;
    initialize_matrix(&A1, nn->W1.rows, X_T.cols);

    // Second Layer
    Matrix Z2;
    initialize_matrix(&Z2, nn->W2.rows, X_T.cols);
    Matrix A2;
    initialize_matrix(&A2, nn->W2.rows, X_T.cols);

    // Output Layer
    Matrix ZOutput;
    initialize_matrix(&ZOutput, nn->WOutput.rows, X_T.cols);
    Matrix AOutput;
    initialize_matrix(&AOutput, nn->WOutput.rows, X_T.cols);

    // Initialize Vectors for Y and Y_hat
    Vector Y_true;
    initialize_vector(&Y_true, X_T.cols);
    Vector Y_hat;
    initialize_vector(&Y_hat, X_T.cols);

    // Forward Propagation
    printf("Forward Propagating Neural Network:\n");
    forward_propagation(&X_T,
            &(nn->W1), &(nn->b1),
	    &(nn->W2), &(nn->b2),
            &(nn->WOutput), &(nn->bOutput),
	    &Z1, &A1,
            &Z1, &A1,
	    &ZOutput, &AOutput);

////////////////////////////////////////////////////////////////////////
// Test GPU Forward Propagation Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");

    printf("Testing GPU Forward Propagation Function:\n");

    // Initialize Neural Network on device
    printf("Initializing Neural Network on device:\n");
    NeuralNetwork_GPU d_nn_value;
    initialize_neural_network_on_device(&d_nn_value,
		    		       NUM_NEURONS_INPUT,
				       NUM_NEURONS_HIDDEN_1,
				       NUM_NEURONS_HIDDEN_2,
				       NUM_NEURONS_OUTPUT);

    // Create pointer to Neural Network on device
    NeuralNetwork_GPU *d_nn = &d_nn_value;

    // Define number of threads per block, number of blocks, and shared memory size
    dim3 threads_per_block_fp (32, 32, 1); // A 16 x 16 block threads
    int N_fp = NUM_ROWS_TRAIN;
    dim3 number_of_blocks_fp ((N_fp / threads_per_block_fp.x) + 1, (N_fp / threads_per_block_fp.y) + 1, 1);
    int sharedMemSize_fp = sizeof(float) * 32 * 32;


    // Copy Neural Network to device
    printf("Copying Neural Network to device:\n");
    copy_neural_network_to_device(&nn_value, &d_nn_value);

    // Preview Input Layer Weights
    printf("Previewing Input Layer Weights:\n");
    preview_matrix_GPU(&(d_nn->W1), 2);

    // Preview Input Layer Biases
    printf("Previewing Input Layer Biases:\n");
    preview_vector_GPU(&(d_nn->b1), 2);

    // Preview Hidden Layer 1 Weights
    printf("Previewing Hidden Layer 1 Weights:\n");
    preview_matrix_GPU(&(d_nn->W2), 2);

    // Preview Hidden Layer 1 Biases
    printf("Previewing Hidden Layer 1 Biases:\n");
    preview_vector_GPU(&(d_nn->b2), 2);

    // Preview Hidden Layer 2 Weights
    printf("Previewing Hidden Layer 2 Weights:\n");
    preview_matrix_GPU(&(d_nn->WOutput), 2);

    // Preview Hidden Layer 2 Biases
    printf("Previewing Hidden Layer 2 Biases:\n");
    preview_vector_GPU(&(d_nn->bOutput), 2);

    // Initialize Matrix d_X_T on device
    printf("Initializing Matrix d_X_T on device:\n");
    Matrix_GPU d_X_T;
    initialize_matrix_on_device(&d_X_T, X_train.cols, X_train.rows);

    // Transpose Matrix d_X_train
    printf("Transposing Matrix d_X_train on device:\n");
    transpose_matrix_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_X_train.data,
		                                                d_X_T.data,
								d_X_train.rows,
								d_X_train.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // Initialize Matrix d_Y_T on device
    printf("Initializing Matrix d_Y_T on device:\n");
    Matrix_GPU d_Y_T;
    initialize_matrix_on_device(&d_Y_T, Y_train.cols, Y_train.rows);

    // Transpose Matrix d_Y_train
    printf("Transposing Matrix d_Y_train on device:\n");
    transpose_matrix_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_Y_train.data,
		                                                d_Y_T.data,
								d_Y_train.rows,
								d_Y_train.cols);

    cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

    // First Layer
    Matrix_GPU d_Z1;
    initialize_matrix_on_device(&d_Z1, d_nn->W1.rows, d_X_T.cols);
    Matrix_GPU d_A1;
    initialize_matrix_on_device(&d_A1, d_nn->W1.rows, d_X_T.cols);

    // Second Layer
    Matrix_GPU d_Z2;
    initialize_matrix_on_device(&d_Z2, d_nn->W2.rows, d_X_T.cols);
    Matrix_GPU d_A2;
    initialize_matrix_on_device(&d_A2, d_nn->W2.rows, d_X_T.cols);

    // Output Layer
    Matrix_GPU d_ZOutput;
    initialize_matrix_on_device(&d_ZOutput, d_nn->WOutput.rows, d_X_T.cols);
    Matrix_GPU d_AOutput;
    initialize_matrix_on_device(&d_AOutput, d_nn->WOutput.rows, d_X_T.cols);

    // Initialize Vectors for Y and Y_hat
    Vector d_Y_true;
    initialize_vector_on_device(&d_Y_true, d_X_T.cols);
    Vector d_Y_hat;
    initialize_vector_on_device(&d_Y_hat, d_X_T.cols);

    // Forward Propagation
    forward_propagation_GPU(&d_X_T,
		    	    &d_nn->W1, &d_nn->b1,
			    &d_nn->W2, &d_nn->b2,
			    &d_nn->WOutput, &d_nn->bOutput,
			    &d_Z1, &d_A1,
			    &d_Z2, &d_A2,
			    &d_ZOutput, &d_AOutput,
			    threads_per_block_fp,
			    number_of_blocks_fp,
			    sharedMemSize_fp);

////////////////////////////////////////////////////////////////////////
// Test CPU Predict Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    
    printf("Testing CPU Predict Function:\n");

    // Initialize Matrix Y_Pred on host
    printf("Initializing Matrix Y_Pred on host:\n");
    Matrix Y_Pred;
    initialize_matrix(&Y_Pred, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);

    // Predict
    printf("Predicting using CPU:\n");
    predict(&nn_value, &X_train, &Y_train, &Y_Pred);

    // Free Matrix Y_Pred
    printf("Freeing Matrix Y_Pred on host:\n");
    free_matrix(&Y_Pred);

////////////////////////////////////////////////////////////////////////
// Test GPU Predict Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    
    printf("Testing GPU Predict Function:\n");

    // Initialize Matrix d_Y_Pred on device
    printf("Initializing Matrix d_Y_Pred on device:\n");
    Matrix_GPU d_Y_Pred;
    initialize_matrix_on_device(&d_Y_Pred, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);

    // Predict
    printf("Predicting using GPU:\n");
    predict_GPU(&d_nn_value, &d_X_train, &d_Y_train, &d_Y_Pred,
		threads_per_block_fp, number_of_blocks_fp, sharedMemSize_fp);

////////////////////////////////////////////////////////////////////////
// Test Preview Predictions Function
    printf("\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    printf("--------------------------------------------------\n");
    
    printf("Testing Preview Predictions Function:\n");

    // Initialize Matrix Y_Pred on host
    printf("Initializing Matrix Y_Pred on host:\n");
    Matrix Y_Pred2;
    initialize_matrix(&Y_Pred2, NUM_ROWS_TRAIN, NUM_NEURONS_OUTPUT);

    // Copy Matrix Y_Pred from device
    printf("Copying Matrix Y_Pred from device:\n");
    copy_matrix_to_host(&Y_Pred2, &d_Y_Pred);

    // Preview Predictions
    printf("Previewing Predictions:\n");
    preview_predictions(&X_train, &Y_Pred2, 28, 28, 5);

    // Free Matrix Y_Pred2
    printf("Freeing Matrix Y_Pred2 on host:\n");
    free_matrix(&Y_Pred2);

    // Free Matrix d_Y_Pred
    printf("Freeing Matrix d_Y_Pred on device:\n");
    free_matrix_on_device(&d_Y_Pred);

// ////////////////////////////////////////////////////////////////////////
// // Test CPU Update Parameters Function
//     printf("\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
//     
//     printf("Testing CPU Update Parameters Function:\n");
// 
//     // Initialize Matrix W1_up on host
//     printf("Initializing Matrix W1_up on host:\n");
//     Matrix W1_up;
//     initialize_matrix(&W1_up, 2, 3);
//     random_matrix(&W1_up);
//     // Initialize Vector b1_up on host
//     printf("Initializing Vector b1_up on host:\n");
//     Vector b1_up;
//     initialize_vector(&b1_up, 2);
//     random_vector(&b1_up);
// 
//     // Initialize Matrix W2_up on host
//     printf("Initializing Matrix W2_up on host:\n");
//     Matrix W2_up;
//     initialize_matrix(&W2_up, 2, 3);
//     random_matrix(&W2_up);
//     preview_matrix(&W2_up, 2);
//     // Initialize Vector b2_up on host
//     printf("Initializing Vector b2_up on host:\n");
//     Vector b2_up;
//     initialize_vector(&b2_up, 2);
//     random_vector(&b2_up);
//     preview_vector(&b2_up, 2);
// 
//     // Initialize Matrix WOutput_up on host
//     printf("Initializing Matrix WOutput_up on host:\n");
//     Matrix WOutput_up;
//     initialize_matrix(&WOutput_up, 2, 3);
//     random_matrix(&WOutput_up);
//     preview_matrix(&WOutput_up, 2);
//     // Initialize Vector bOutput_up on host
//     printf("Initializing Vector bOutput_up on host:\n");
//     Vector bOutput_up;
//     initialize_vector(&bOutput_up, 2);
//     random_vector(&bOutput_up);
//     preview_vector(&bOutput_up, 2);
// 
//     // Initialize Matrix dW1_up on host
//     printf("Initializing Matrix dW1_up on host:\n");
//     Matrix dW1_up;
//     initialize_matrix(&dW1_up, 2, 3);
//     random_matrix(&dW1_up);
//     preview_matrix(&dW1_up, 2);
//     // Initialize float db1_up on host
//     printf("Initializing Vector db1_up on host:\n");
//     float db1_up = 200;
//     printf("%f\n", db1_up);
// 
//     // Initialize Matrix dW2_up on host
//     printf("Initializing Matrix dW2_up on host:\n");
//     Matrix dW2_up;
//     initialize_matrix(&dW2_up, 2, 3);
//     random_matrix(&dW2_up);
//     preview_matrix(&dW2_up, 2);
//     // Initialize float db2_up on host
//     printf("Initializing Vector db2_up on host:\n");
//     float db2_up = 200;
//     printf("%f\n", db2_up);
// 
//     // Initialize Matrix dWOutput_up on host
//     printf("Initializing Matrix dWOutput_up on host:\n");
//     Matrix dWOutput_up;
//     initialize_matrix(&dWOutput_up, 2, 3);
//     random_matrix(&dWOutput_up);
//     preview_matrix(&dWOutput_up, 2);
//     // Initialize float dbOutput_up on host
//     printf("Initializing Vector dbOutput_up on host:\n");
//     float dbOutput_up = 200;
//     printf("%f\n", dbOutput_up);
// 
//     // Update Parameters
//     printf("Updating Parameters:\n");
//     float learning_rate_up = 0.01;
//     update_parameters(&W1_up, &b1_up,
// 		      &W2_up, &b2_up,
// 		      &WOutput_up, &bOutput_up,
// 		      &dW1_up, db1_up,
// 		      &dW2_up, db2_up,
// 		      &dWOutput_up, dbOutput_up,
// 		      learning_rate_up);
// 
//     // Preview Updated Parameters
//     printf("Previewing Updated Parameters:\n");
//     printf("Previewing W1_up:\n");
//     preview_matrix(&W1_up, 2);
//     printf("Previewing b1_up:\n");
//     preview_vector(&b1_up, 2);
//     printf("Previewing W2_up:\n");
//     preview_matrix(&W2_up, 2);
//     printf("Previewing b2_up:\n");
//     preview_vector(&b2_up, 2);
//     printf("Previewing WOutput_up:\n");
//     preview_matrix(&WOutput_up, 2);
//     printf("Previewing bOutput_up:\n");
//     preview_vector(&bOutput_up, 2);
// 
// ////////////////////////////////////////////////////////////////////////
// // Testing GPU Update Parameters Function
//     printf("\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
//     
//     printf("Testing GPU Update Parameters Function:\n");
// 
//     // Initialize Matrix d_W1_up on device
//     printf("Initializing Matrix d_W1_up on device:\n");
//     Matrix_GPU d_W1_up;
//     initialize_matrix_on_device(&d_W1_up, 2, 3);
//     copy_matrix_to_device(&W1_up, &d_W1_up);
//     // Initialize Vector d_b1_up on device
//     printf("Initializing Vector d_b1_up on device:\n");
//     Vector d_b1_up;
//     initialize_vector_on_device(&d_b1_up, 2);
//     copy_vector_to_device(&b1_up, &d_b1_up);
// 
//     // Initialize Matrix d_W2_up on device
//     printf("Initializing Matrix d_W2_up on device:\n");
//     Matrix_GPU d_W2_up;
//     initialize_matrix_on_device(&d_W2_up, 2, 3);
//     copy_matrix_to_device(&W2_up, &d_W2_up);
// 
//     // Initialize Vector d_b2_up on device
//     printf("Initializing Vector d_b2_up on device:\n");
//     Vector d_b2_up;
//     initialize_vector_on_device(&d_b2_up, 2);
//     copy_vector_to_device(&b2_up, &d_b2_up);
// 
//     // Initialize Matrix d_WOutput_up on device
//     printf("Initializing Matrix d_WOutput_up on device:\n");
//     Matrix_GPU d_WOutput_up;
//     initialize_matrix_on_device(&d_WOutput_up, 2, 3);
//     copy_matrix_to_device(&WOutput_up, &d_WOutput_up);
//     // Initialize Vector d_bOutput_up on device
//     printf("Initializing Vector d_bOutput_up on device:\n");
//     Vector d_bOutput_up;
//     initialize_vector_on_device(&d_bOutput_up, 2);
//     copy_vector_to_device(&bOutput_up, &d_bOutput_up);
// 
//     // Initialize Matrix d_dW1_up on device
//     printf("Initializing Matrix d_dW1_up on device:\n");
//     Matrix_GPU d_dW1_up;
//     initialize_matrix_on_device(&d_dW1_up, 2, 3);
//     copy_matrix_to_device(&dW1_up, &d_dW1_up);
//     // Initialize float d_db1_up on device
//     printf("Initializing float d_db1_up on device:\n");
//     float d_db1_up;
//     cudaMemcpy(&d_db1_up, &db1_up, sizeof(float), cudaMemcpyHostToDevice);
// 
//     // Initialize Matrix d_dW2_up on device
//     printf("Initializing Matrix d_dW2_up on device:\n");
//     Matrix_GPU d_dW2_up;
//     initialize_matrix_on_device(&d_dW2_up, 2, 3);
//     copy_matrix_to_device(&dW2_up, &d_dW2_up);
//     // Initialize float d_db2_up on device
//     printf("Initializing float d_db2_up on device:\n");
//     float d_db2_up;
//     cudaMemcpy(&d_db2_up, &db2_up, sizeof(float), cudaMemcpyHostToDevice);
// 
//     // Initialize Matrix d_dWOutput_up on device
//     printf("Initializing Matrix d_dWOutput_up on device:\n");
//     Matrix_GPU d_dWOutput_up;
//     initialize_matrix_on_device(&d_dWOutput_up, 2, 3);
//     copy_matrix_to_device(&dWOutput_up, &d_dWOutput_up);
//     // Initialize float d_dbOutput_up on device
//     printf("Initializing float d_dbOutput_up on device:\n");
//     float d_dbOutput_up;
//     cudaMemcpy(&d_dbOutput_up, &dbOutput_up, sizeof(float), cudaMemcpyHostToDevice);
// 
// 
//     // Update Parameters
//     printf("Updating Parameters:\n");
//     dim3 threads_per_block5 (16, 16, 1); // A 16 x 16 block threads
//     // Remember the testing matrices are 2 x 3
//     int N5 = 2;
//     dim3 number_of_blocks5 ((N5 / threads_per_block5.x) + 1, (N5 / threads_per_block5.y) + 1, 1);
//     update_parameters_GPU <<< threads_per_block5, number_of_blocks5 >>> (d_W1_up.data, d_b1_up.data,
// 		                                                d_W2_up.data, d_b2_up.data,
// 								d_WOutput_up.data, d_bOutput_up.data,
// 								d_dW1_up.data, d_db1_up,
// 								d_dW2_up.data, d_db2_up,
// 								d_dWOutput_up.data, d_dbOutput_up,
// 								d_W1_up.rows, d_W1_up.cols,
// 								d_b1_up.rows,
// 								d_W2_up.rows, d_W2_up.cols,
// 								d_b2_up.rows,
// 								d_WOutput_up.rows, d_WOutput_up.cols,
// 								d_bOutput_up.rows,
// 								learning_rate_up);
// 
//     cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
// 
//     // Preview Updated Parameters
//     printf("Previewing Updated Parameters:\n");
//     printf("Previewing d_W1_up:\n");
//     preview_matrix_GPU(&d_W1_up, 2);
//     printf("Previewing d_b1_up:\n");
//     preview_vector_GPU(&d_b1_up, 2);
//     printf("Previewing d_W2_up:\n");
//     preview_matrix_GPU(&d_W2_up, 2);
//     printf("Previewing d_b2_up:\n");
//     preview_vector_GPU(&d_b2_up, 2);
//     printf("Previewing d_WOutput_up:\n");
//     preview_matrix_GPU(&d_WOutput_up, 2);
//     printf("Previewing d_bOutput_up:\n");
//     preview_vector_GPU(&d_bOutput_up, 2);
// 
// ////////////////////////////////////////////////////////////////////////
// // Test CPU Backpropagation Function
//     printf("\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
// 
//     // Normalize X_train
//     printf("Normalizing X_train:\n");
//     normalize_matrix(&X_train, 0, 255);
//     
//     // Vectors/Matrices Needed for Calculation of Output Layer Gradients
//     // dZOutput = AOutput - Y_T
//     Matrix dZOutput;
//     initialize_matrix(&dZOutput, ZOutput.rows, ZOutput.cols);
//     // dWOutput = 1/m * matmul(dZOutput, A2_T)
//     Matrix dWOutput;
//     initialize_matrix(&dWOutput, nn->WOutput.rows, nn->WOutput.cols);
//     Matrix A2_T;
//     initialize_matrix(&A2_T, A2.cols, A2.rows);
//     // dbOutput = 1/m * sum(dZOutput)
//     float dbOutput;
// 
//     // Vectors/Matrices Needed for Calculation of Second Layer Gradients
//     // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_derivative(Z2)
//     Matrix dZ2;
//     initialize_matrix(&dZ2, Z2.rows, Z2.cols);
//     Matrix WOutput_T;
//     initialize_matrix(&WOutput_T, nn->WOutput.cols, nn->WOutput.rows);
//     Matrix WOutput_dZOutput; // Product of WOutput_T and dZOutput
//     initialize_matrix(&WOutput_dZOutput, WOutput_T.rows, dZOutput.cols);
//     // dW2 = 1/m * matmul(dZ2, A1_T)
//     Matrix dW2;
//     initialize_matrix(&dW2, nn->W2.rows, nn->W2.cols);
//     Matrix A1_T;
//     initialize_matrix(&A1_T, A1.cols, A1.rows);
//     // db2 = 1/m * sum(dZ2)
//     float db2;
// 
//     // Vectors/Matrices Needed for Calculation of First Layer Gradients
//     // dZ1 = matmul(W2_T, dZ2) * ReLU_deriv(Z1)
//     Matrix dZ1;
//     initialize_matrix(&dZ1, Z1.rows, Z1.cols);
//     Matrix W2_T;
//     initialize_matrix(&W2_T, nn->W2.cols, nn->W2.rows);
//     Matrix W2_dZ2; // Product of W2_T and dZ2
//     initialize_matrix(&W2_dZ2, W2_T.rows, dZ2.cols);
//     // dW1 = 1/m * matmul(dZ1, X_T)
//     Matrix dW1;
//     initialize_matrix(&dW1, nn->W1.rows, nn->W1.cols);
//     // db1 = 1/m * sum(dZ1)
//     float db1;
// 
//     // Training Rounds
//     int epochs = 1;
//     float learning_rate = 0.01;
// 
//     // Loop over the epochs
//     for (int epoch = 0; epoch < epochs; epoch++) {
// 	printf("Epoch %d:\n", epoch);
// 
// 	// Forward Propagation
// 	forward_propagation(&X_T,
// 			&(nn->W1), &(nn->b1),
// 		     	&(nn->W2), &(nn->b2),
// 			&(nn->WOutput), &(nn->bOutput),
// 			&Z1, &A1,
// 		     	&Z2, &A2,
// 		        &ZOutput, &AOutput);
// 
// 	// Backward Propagation
// 	backward_propagation(&X_T, &Y_T,
// 			     &(nn->W1), &(nn->b1),
// 		      	     &(nn->W2), &(nn->b2),
// 			     &(nn->WOutput), &(nn->bOutput),
// 			     &Z1, &A1,
// 		      	     &Z2, &A2,
// 			     &ZOutput, &AOutput,
// 			     &dW1, &db1,
// 		      	     &dW2, &db2,
// 			     &dWOutput, &dbOutput,
// 			     &dZ1, &dZ2, &dZOutput,
// 		      	     &WOutput_T,
// 		             &WOutput_dZOutput,
// 		      	     &W2_T,
// 		      	     &W2_dZ2,
// 			     &A2_T, &A1_T, &X_train);
// 
// 	// Update Parameters
// 	update_parameters(&(nn->W1), &(nn->b1),
// 		   	  &(nn->W2), &(nn->b2),
// 		          &(nn->WOutput), &(nn->bOutput),
// 		          &dW1, db1,
// 		          &dW2, db2,
// 		          &dWOutput, dbOutput,
// 		          learning_rate);
// 
// 	// Get Predictions
// 	argmax(&Y_T, &Y_true);
// 	argmax(&AOutput, &Y_hat);
// 
// 	// Calculate Accuracy
// 	calculate_accuracy(&Y_true, &Y_hat);
//     }
// 
// ////////////////////////////////////////////////////////////////////////
// // Testing GPU Backpropagation Function
//     printf("\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
// 
//     printf("Testing GPU Backpropagation Function:\n");
// 
//     // Vectors/Matrices Needed for Calculation of Output Layer Gradients
//     // dZOutput = AOutput - Y_T
//     Matrix_GPU d_dZOutput;
//     initialize_matrix_on_device(&d_dZOutput, dZOutput.rows, dZOutput.cols);
//     copy_matrix_to_device(&dZOutput, &d_dZOutput);
//     // dWOutput = 1/m * matmul(dZOutput, A2_T)
//     Matrix_GPU d_dWOutput;
//     initialize_matrix_on_device(&d_dWOutput, nn->WOutput.rows, nn->WOutput.cols);
//     copy_matrix_to_device(&dWOutput, &d_dWOutput);
//     Matrix_GPU d_A2_T;
//     initialize_matrix_on_device(&d_A2_T, A2.cols, A2.rows);
//     copy_matrix_to_device(&A2_T, &d_A2_T);
//     // dbOutput = 1/m * sum(dZOutput)
//     float d_dbOutput;
//     cudaMalloc((void **)&d_dbOutput, sizeof(float));
// 
//     // Vectors/Matrices Needed for Calculation of Second Layer Gradients
//     // dZ2 = matmul(WOutput_T, dZOutput) * ReLU_derivative(Z2)
//     Matrix_GPU d_dZ2;
//     initialize_matrix_on_device(&d_dZ2, dZ2.rows, dZ2.cols);
//     copy_matrix_to_device(&dZ2, &d_dZ2);
//     Matrix_GPU d_WOutput_T;
//     initialize_matrix_on_device(&d_WOutput_T, nn->WOutput.cols, nn->WOutput.rows);
//     copy_matrix_to_device(&WOutput_T, &d_WOutput_T);
//     Matrix_GPU d_WOutput_dZOutput; // Product of WOutput_T and dZOutput
//     initialize_matrix_on_device(&d_WOutput_dZOutput, d_WOutput_T.rows, d_dZOutput.cols);
//     copy_matrix_to_device(&WOutput_dZOutput, &d_WOutput_dZOutput);
//     // dW2 = 1/m * matmul(dZ2, A1_T)
//     Matrix_GPU d_dW2;
//     initialize_matrix_on_device(&d_dW2, nn->W2.rows, nn->W2.cols);
//     copy_matrix_to_device(&dW2, &d_dW2);
//     Matrix_GPU d_A1_T;
//     initialize_matrix_on_device(&d_A1_T, A1.cols, A1.rows);
//     copy_matrix_to_device(&A1_T, &d_A1_T);
//     // db2 = 1/m * sum(dZ2)
//     float d_db2;
//     cudaMalloc((void **)&d_db2, sizeof(float));
// 
//     // Vectors/Matrices Needed for Calculation of First Layer Gradients
//     // dZ1 = matmul(W2_T, dZ2) * ReLU_deriv(Z1)
//     Matrix_GPU d_dZ1;
//     initialize_matrix_on_device(&d_dZ1, dZ1.rows, dZ1.cols);
//     copy_matrix_to_device(&dZ1, &d_dZ1);
//     Matrix_GPU d_W2_T;
//     initialize_matrix_on_device(&d_W2_T, nn->W2.cols, nn->W2.rows);
//     copy_matrix_to_device(&W2_T, &d_W2_T);
//     Matrix_GPU d_W2_dZ2; // Product of W2_T and dZ2
//     initialize_matrix_on_device(&d_W2_dZ2, d_W2_T.rows, d_dZ2.cols);
//     copy_matrix_to_device(&W2_dZ2, &d_W2_dZ2);
//     // dW1 = 1/m * matmul(dZ1, X_T)
//     Matrix_GPU d_dW1;
//     initialize_matrix_on_device(&d_dW1, nn->W1.rows, nn->W1.cols);
//     copy_matrix_to_device(&dW1, &d_dW1);
//     // db1 = 1/m * sum(dZ1)
//     float d_db1;
//     cudaMalloc((void **)&d_db1, sizeof(float));
// 
//     // Training Rounds
//     int d_epochs = 1;
//     float d_learning_rate = 0.01;
//     for (int epoch = 0; epoch < d_epochs; epoch++) {
// 	printf("Epoch %d:\n", epoch);
// 
// 	// Forward Propagation
// 	forward_propagation_GPU(&d_X_T,
// 		    	       &d_nn->W1, &d_nn->b1,
// 			       &d_nn->W2, &d_nn->b2,
// 			       &d_nn->WOutput, &d_nn->bOutput,
// 			       &d_Z1, &d_A1,
// 			       &d_Z2, &d_A2,
// 			       &d_ZOutput, &d_AOutput,
// 			       threads_per_block_fp,
// 			       number_of_blocks_fp,
// 			       sharedMemSize_fp);
// 
// 	cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
// 
// 	// Backward Propagation
// 	backward_propagation_GPU(&d_X_T, &d_Y_T,
// 				 &d_nn->W1, &d_nn->b1,
// 		      	         &d_nn->W2, &d_nn->b2,
// 			         &d_nn->WOutput, &d_nn->bOutput,
// 			         &d_Z1, &d_A1,
// 		      	         &d_Z2, &d_A2,
// 			         &d_ZOutput, &d_AOutput,
// 			         &d_dW1, &d_db1,
// 		      	         &d_dW2, &d_db2,
// 			         &d_dWOutput, &d_dbOutput,
// 			         &d_dZ1, &d_dZ2, &d_dZOutput,
// 		      	         &d_WOutput_T,
// 		                 &d_WOutput_dZOutput,
// 		      	         &d_W2_T,
// 		      	         &d_W2_dZ2,
// 			         &d_A2_T, &d_A1_T, &d_X_train,
// 				 threads_per_block_fp,
// 				 number_of_blocks_fp,
// 				 sharedMemSize_fp);
// 
// 	cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
// 
// 	// Update Parameters
// 	update_parameters_GPU <<< threads_per_block5, number_of_blocks5 >>> (d_nn->W1.data, d_nn->b1.data,
// 		                                                        d_nn->W2.data, d_nn->b2.data,
// 								d_nn->WOutput.data, d_nn->bOutput.data,
// 								d_dW1.data, d_db1,
// 								d_dW2.data, d_db2,
// 								d_dWOutput.data, d_dbOutput,
// 								d_nn->W1.rows, d_nn->W1.cols,
// 								d_nn->b1.rows,
// 								d_nn->W2.rows, d_nn->W2.cols,
// 								d_nn->b2.rows,
// 								d_nn->WOutput.rows, d_nn->WOutput.cols,
// 								d_nn->bOutput.rows,
// 								d_learning_rate);
// 
// 	cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
// 
// 	// Get Predictions
// 	argmax_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_Y_T.data,
// 			                                              d_Y_true.data,
// 		                                                      d_Y_T.rows,
// 								      d_Y_T.cols);
// 	cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
// 	argmax_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_AOutput.data,
// 			                                              d_Y_hat.data,
// 		                                                      d_AOutput.rows,
// 								      d_AOutput.cols);
// 	cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
// 
// 	// Initialize float d_accuracy
// 	float *d_accuracy;
// 	cudaMalloc((void **)&d_accuracy, sizeof(float));
// 
// 	// Calculate Accuracy
// 	calculate_accuracy_GPU <<< number_of_blocks_fp, threads_per_block_fp >>> (d_Y_true.data,
// 			                                                          d_Y_hat.data,
// 		                                                                  d_Y_true.rows,
// 								                  d_accuracy);
// 	cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding
// 
// 	// Copy float d_accuracy to host
// 	float h_accuracy;
// 	cudaMemcpy(&h_accuracy, d_accuracy, sizeof(float), cudaMemcpyDeviceToHost);
// 	printf("Accuracy: %f\n", h_accuracy);
// 
//     }
// 
// ////////////////////////////////////////////////////////////////////////
// // Test CPU Training Function
//     printf("\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
// 
//     printf("Testing CPU Training Function:\n");
// 
//     // Initialize Neural Network
//     NeuralNetwork nn2;
//     initialize_neural_network(&nn2,
// 			      NUM_NEURONS_INPUT,
// 			      NUM_NEURONS_HIDDEN_1,
// 			      NUM_NEURONS_HIDDEN_2,
// 			      NUM_NEURONS_OUTPUT);
// 
//     // train(&nn2, &X_train, &Y_train, 30, 0.1);
//     // train(&nn2, &X_train, &Y_train, 2, 0.1);
// 
// ////////////////////////////////////////////////////////////////////////
// // Test GPU Training Function
//     printf("\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
//     printf("--------------------------------------------------\n");
// 
//     printf("Testing GPU Training Function:\n");
// 
//     // Initialize Neural Network on host
//     NeuralNetwork nn3;
//     initialize_neural_network(&nn3,
// 		    		       NUM_NEURONS_INPUT,
// 				       NUM_NEURONS_HIDDEN_1,
// 				       NUM_NEURONS_HIDDEN_2,
// 				       NUM_NEURONS_OUTPUT);
// 
//     // Initialize Neural Network on device
//     NeuralNetwork_GPU d_nn3;
//     initialize_neural_network_on_device(&d_nn3,
// 		    		       NUM_NEURONS_INPUT,
// 				       NUM_NEURONS_HIDDEN_1,
// 				       NUM_NEURONS_HIDDEN_2,
// 				       NUM_NEURONS_OUTPUT);
// 
//     // Copy Neural Network to device
//     printf("Copying Neural Network to device:\n");
//     copy_neural_network_to_device(&nn3, &d_nn3);
// 
//     // Train
//     // train_GPU(&d_nn3, &d_X_train, &d_Y_train, 2, 0.1,
//     // 	      threads_per_block_fp, number_of_blocks_fp, sharedMemSize_fp);
// 
////////////////////////////////////////////////////////////////////////
    // Free CPU Neural Network
    free_neural_network(&nn_value);
    free_matrix(&X_T);
    free_matrix(&Y_T);
    free_matrix(&Z1);
    free_matrix(&A1);
    free_matrix(&Z2);
    free_matrix(&A2);
    free_matrix(&ZOutput);
    free_matrix(&AOutput);
    free_vector(&Y_true);
    free_vector(&Y_hat);

    // free_matrix(&dZOutput);
    // free_matrix(&dWOutput);
    // free_matrix(&A2_T);
    // free_matrix(&dZ2);
    // free_matrix(&WOutput_T);
    // free_matrix(&WOutput_dZOutput);
    // free_matrix(&dW2);
    // free_matrix(&A1_T);
    // free_matrix(&dZ1);
    // free_matrix(&W2_T);
    // free_matrix(&W2_dZ2);
    // free_matrix(&dW1);

////////////////////////////////////////////////////////////////////////
    // Free GPU Neural Network
    free_neural_network_on_device(&d_nn_value);
    free_matrix_on_device(&d_X_T);
    free_matrix_on_device(&d_Y_T);
    free_matrix_on_device(&d_Z1);
    free_matrix_on_device(&d_A1);
    free_matrix_on_device(&d_Z2);
    free_matrix_on_device(&d_A2);
    free_matrix_on_device(&d_ZOutput);
    free_matrix_on_device(&d_AOutput);
    free_vector_on_device(&d_Y_true);
    free_vector_on_device(&d_Y_hat);

    // free_matrix_on_device(&d_dZOutput);
    // free_matrix_on_device(&d_dWOutput);
    // free_matrix_on_device(&d_A2_T);
    // free_matrix_on_device(&d_dZ2);
    // free_matrix_on_device(&d_WOutput_T);
    // free_matrix_on_device(&d_WOutput_dZOutput);
    // free_matrix_on_device(&d_dW2);
    // free_matrix_on_device(&d_A1_T);
    // free_matrix_on_device(&d_dZ1);
    // free_matrix_on_device(&d_W2_T);
    // free_matrix_on_device(&d_W2_dZ2);
    // free_matrix_on_device(&d_dW1);

////////////////////////////////////////////////////////////////////////
// Final Freeing of Memory

    // Free X_train
    printf("Freeing X_train on host:\n");
    free_matrix(&X_train);

    // Free d_X_train
    printf("Freeing d_X_train on device:\n");
    free_matrix_on_device(&d_X_train);

    // Free Y_train
    printf("Freeing Y_train on host:\n");
    free_matrix(&Y_train);

    // Free d_Y_train
    printf("Freeing d_Y_train on device:\n");
    free_matrix_on_device(&d_Y_train);

    return 0;
}
