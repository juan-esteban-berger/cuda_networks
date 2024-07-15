import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

####################################################################
# Load Data
print("Loading Features...")
X_train = pd.read_csv('data/X_train.csv', header=None).to_numpy()
X_test = pd.read_csv('data/X_test.csv', header=None).to_numpy()
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

print("Loading Targets...")
Y_train = pd.read_csv('data/Y_train.csv', header=None).to_numpy()
Y_test = pd.read_csv('data/Y_test.csv', header=None).to_numpy()
print(f"Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")

####################################################################
# Get Number of Training Examples
m = X_train.shape[0]

####################################################################
# Transpose Data
X_train = X_train.T
X_test = X_test.T

Y_train = Y_train.T
Y_test = Y_test.T

####################################################################
# Normalize X values
X_test = X_test / 255.
X_train = X_train / 255.

####################################################################
# Function to Display Image
def display_image(X, index):
    image = X[:, index].reshape(28, 28)
    for row in image:
        for pixel in row:
            if pixel == 0:
                print("  ", end="")
            else:
                print("##", end="")
        print()

####################################################################
# Activation Function Classes
class Sigmoid():
    def function(self, Z):
        return 1 / (1 + np.exp(-Z))

    def derivative(self, Z):
        return self.function(Z) * (1 - self.function(Z))

class Softmax():
    def function(self, Z):
        e_Z = np.exp(Z - np.max(Z))
        return e_Z / (e_Z.sum(axis=0) + 1e-8)

####################################################################
# Loss Function Classes
class CatCrossEntropy():
    def function(self, Y, Y_hat):
        return -np.sum(Y * np.log(Y_hat + 1e-8))

####################################################################
# Layer Class
class Layer:
    def __init__(self, input_num, output_num, activation):
        self.W = np.random.rand(output_num, input_num) - 0.5
        self.b = np.random.rand(output_num, 1) - 0.5
       
        self.Z = None
        self.A = None

        self.dZ = None
        self.dW = None
        self.db = None

        self.activation = activation

####################################################################
# Neural Network Class
class NeuralNetwork():
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

    def backward(self, X, Y, loss):
        m = X.shape[1]

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            if (i == len(self.layers) - 1 and
               isinstance(layer.activation, Softmax) and
               isinstance(loss, CatCrossEntropy)):
                layer.dZ = layer.A - Y
            else:
                next_layer = self.layers[i + 1]
                layer.dZ = (next_layer.W.T.dot(next_layer.dZ) *
                            layer.activation.derivative(layer.Z))

            if i != 0:
                prev_layer = self.layers[i - 1]
                layer.dW = 1 / m * layer.dZ.dot(prev_layer.A.T)

            else:
                layer.dW = 1 / m * layer.dZ.dot(X.T)

            layer.db = 1 / m * np.sum(layer.dZ, axis=1, keepdims=True)

    def update_params(self, learning_rate):
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

    def get_accuracy(self, Y):
        predictions = np.argmax(self.layers[-1].A, 0)
        Y_decoded = np.argmax(Y, 0)
        return np.sum(predictions == Y_decoded) / Y_decoded.size

    def train(self, X_train, Y_train, epochs, learning_rate, loss):
        pbar = tqdm(range(epochs), position=0, leave=True)
        for epoch in pbar:
            self.forward(X_train)
            self.backward(X_train, Y_train, loss)
            self.update_params(learning_rate)

            acc = self.get_accuracy(Y_train)
            loss_val = loss.function(Y_train, self.layers[-1].A)
            description = ("Epoch: %d, Accuracy: %f, Loss: %.0f" %
                           (epoch, acc, loss_val))
            pbar.set_description(description)

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.layers[-1].A, axis=0)

    def save_config(self, filepath):
        config = []
        for layer in self.layers:
            input_num = layer.W.shape[1]
            output_num = layer.W.shape[0]
            activation = layer.activation.__class__.__name__
            config.append(','.join([str(input_num),
                                    str(output_num),
                                    activation]))
        config_str = "\n".join(config)

        with open(filepath, 'w') as f:
            f.write(config_str)


    def save_weights(self, filepath):
        with open(filepath, 'w') as f:
            for layer in self.layers:
                np.savetxt(f, [layer.W.flatten()], delimiter=',', fmt='%.8f')
                np.savetxt(f, [layer.b.flatten()], delimiter=',', fmt='%.8f')

    def load_config(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        self.layers = []
        for line in lines:
            input_num, output_num, activation = line.strip().split(',')
            input_num = int(input_num)
            output_num = int(output_num)
            if activation == 'Sigmoid':
                activation = Sigmoid()
            elif activation == 'Softmax':
                activation = Softmax()
            self.add_layer(Layer(input_num, output_num, activation))

    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            i = 0
            for layer in self.layers:
                layer.W = np.fromstring(lines[i].strip(),
                                        sep=',').reshape(layer.W.shape)
                i += 1
                layer.b = np.fromstring(lines[i].strip(),
                                        sep=',').reshape(layer.b.shape)
                i += 1

####################################################################
nn = NeuralNetwork()

nn.add_layer(Layer(784, 200, Sigmoid()))
nn.add_layer(Layer(200, 200, Sigmoid()))
nn.add_layer(Layer(200, 10, Softmax()))

print("Training...")
nn.train(X_train,
         Y_train,
         # epochs=1000,
         epochs=70,
         learning_rate=0.1,
         loss=CatCrossEntropy())

print("Saving Model...")
nn.save_config("models/python_config.csv")
nn.save_weights("models/python_weights.csv")

####################################################################
nn_loaded = NeuralNetwork()

print("Loading Model...")
nn_loaded.load_config("models/python_config.csv")
nn_loaded.load_weights("models/python_weights.csv")

print("Testing...")
pred = nn_loaded.predict(X_test)
acc = nn_loaded.get_accuracy(Y_test)
print(f"Accuracy: {acc}")

####################################################################
print("Displaying 5 Random Images...")
for i in range(5):
    random_index = np.random.randint(0, X_test.shape[1])

    pred_val = pred[random_index]

    Y_decoded = np.argmax(Y_test, 0)
    y_val = Y_decoded[random_index]

    print(f"Predicted: {pred_val}, True: {y_val}")
    display_image(X_test, random_index)
