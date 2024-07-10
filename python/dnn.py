import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

######################################################################
# Load Data
X_train = pd.read_csv('data/X_train.csv', header=None).values
X_test = pd.read_csv('data/X_test.csv', header=None).values
Y_train = pd.read_csv('data/Y_train.csv', header=None).values
Y_test = pd.read_csv('data/Y_test.csv', header=None).values

######################################################################
# Transpose Data
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T

######################################################################
# Activation Function Classes
class ReLu:
    def __init__(self):
        pass

    def function(self, Z):
        return np.maximum(Z, 0)

    def derivative(self, Z):
        return Z > 0

class softmax:
    def __init__(self):
        pass

    def function(self, Z):
        return np.exp(Z) /np.sum(np.exp(Z), axis=0)

    def derivative(self, Z):
        S = self.function(Z)
        return S * (1 - S)

######################################################################
# Loss Functions
### Need to add categorical cross entropy... (do not skip steps)
    ### Categorical Cross Entropy eventually leads to A - Y...
    ### Do it from scratch...

######################################################################
# Layer Class
class Layer:
    def __init__(self, input_num, output_num, activation):
        self.W = np.random.rand(output_num, input_num) - 0.5
        self.b = np.zeros((output_num, 1)) - 0.5
       
        self.Z = None
        self.A = None

        self.dZ = None
        self.dW = None
        self.db = None

        self.activation = activation

######################################################################
# Neural Network Class
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        A = X
        for layer in self.layers:
            layer.Z = layer.W.dot(A) + layer.b
            layer.A = layer.activation.function(layer.Z)
            A = layer.A

    def backward(self, X, Y):
        for i in reversed(range(0, len(self.layers))):
            print(f"Iteration {i}")

    def update_params(self):
        pass

    def train(self, X_train, Y_train, epochs, learning_rate, batch_size):
        for epoch in range(0, epochs):
            for i in tqdm(range(0, X_train.shape[1], batch_size),
                          desc=f'Epoch {epoch+1}/{epochs}'):
                X_batch = X_train[:, i:i+batch_size]
                Y_batch = Y_train[:, i:i+batch_size]

                self.forward(X_batch)
                self.backward(X_batch, Y_batch)
                self.update_params()

    def predict(self):
        pass

######################################################################
# Initialize Neural Network
nn = NeuralNetwork()

# Add Layers
nn.add_layer(Layer(784, 10, ReLu()))
nn.add_layer(Layer(10, 10, ReLu()))
nn.add_layer(Layer(10, 10, softmax()))

# Train Neural Network
nn.train(X_train,
         Y_train,
         epochs=5,
         # epochs=100,
         learning_rate=0.01,
         batch_size=200)
