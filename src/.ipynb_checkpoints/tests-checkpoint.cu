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
    sum_matrix_GPU<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A_sum.data, d_sum, d_A_sum.rows, d_A_sum.cols);
    
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
    int N2 = 64;
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
    softmax_GPU <<< number_of_blocks, threads_per_block >>> (d_A_softmax.data,
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

    return 0;
}
