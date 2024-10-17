# CUDA Neural Networks

Deep neural network implementation from scratch in C++ with CUDA for GPU acceleration. This project focuses on creating a neural network to classify digits using the MNIST dataset.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Memory Check](#memory-check)
- [License](#license)

## Overview

This project implements a deep neural network using C++ and CUDA for GPU acceleration. The main features include:

- Matrix and vector operations optimized for GPU
- Feedforward neural network with customizable architecture
- Training using backpropagation and gradient descent
- MNIST digit classification

## Documentation
For detailed documentation of the project, please visit the [Documentation Page](http://juanberger.com/cuda-networks-docs/).

## System Requirements

This project was developed and tested on a system with the following specifications:

- OS: Arch Linux x86_64
- CPU: Intel i7-9750H (12) @ 4.500GHz
- GPU: NVIDIA GeForce GTX 1650 Mobile / Max-Q
- Memory: 16GB
- NVIDIA Driver Version: 560.35.03
- CUDA Version: 12.6
- CUDA Compilation Tools: Release 12.6, V12.6.77

Please ensure your system meets these requirements or adjust accordingly.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/juan-esteban-berger/cuda_networks.git
   cd cuda_networks
   ```

2. Ensure you have the CUDA Toolkit (version 12.6 or compatible) installed on your system.

3. Install the Google Test library if not already present:
   ```
   sudo pacman -S gtest
   ```

4. Build the project:
   ```
   make
   ```

## Usage

To run the MNIST classifier:

```
make run
```

This will execute the main program, which trains the neural network on the MNIST dataset and displays the training progress and final accuracy.

![Demo GIF](https://github.com/juan-esteban-berger/cuda_networks/blob/main/gifs/demo.gif)

## Running Tests

To run the test suite:

```
make test
```

This will execute all the unit tests for the project.

## Memory Check

To run the compute sanitizer memory check:

```
make memcheck
```

This will use NVIDIA's Compute Sanitizer to check for memory errors in the CUDA code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Modifications

If you encounter any issues running the project, you may need to:

1. Adjust the `NVCCFLAGS` in the Makefile to match your GPU architecture.
2. Ensure the paths to the CUDA Toolkit and Google Test library are correct for your system.
3. If you're using a different CUDA version, you might need to update some CUDA function calls or syntax.
