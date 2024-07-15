import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

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

    def train(self, X_train,
                    Y_train,
                    epochs,
                    learning_rate,
                    loss,
                    history_path):
        accuracy_list = []
        loss_list = []

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

            accuracy_list.append(acc)
            loss_list.append(loss_val)

        df = pd.DataFrame(list(zip(accuracy_list, loss_list)))
        df.to_csv(history_path, index=False, header=False)

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
