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
# Activation Functions
def ReLu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

######################################################################
# Derivatives of Activation Functions
def ReLU_deriv(Z):
    return Z > 0

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
        for layer in self.layers:
            pass

    def backward(self):
        pass

    def update_params(self):
        pass

    def train(self, X_train, Y_train, epochs, learning_rate, batch_size):
        for epoch in range(0, epochs):
            for i in tqdm(range(0, X_train.shape[1], batch_size),
                          desc=f'Epoch {epoch+1}/{epochs}'):
                X_batch = X_train[:, i:i+batch_size]
                Y_batch = Y_train[:, i:i+batch_size]

                self.forward(X_batch)
                self.backward()
                self.update_params()

    def predict(self):
        pass

######################################################################
# Initialize Neural Network
nn = NeuralNetwork()

# Add Layers
nn.add_layer(Layer(784, 10, ReLu))
nn.add_layer(Layer(10, 10, ReLu))
nn.add_layer(Layer(10, 10, softmax))

# Train Neural Network
nn.train(X_train,
         Y_train,
         epochs=5,
         # epochs=100,
         learning_rate=0.01,
         batch_size=200)
