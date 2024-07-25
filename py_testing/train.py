import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from neural_network import Layer, NeuralNetwork
from neural_network import Sigmoid, Softmax, CatCrossEntropy

####################################################################
# Load Data
print("Loading Data...")
X_train = pd.read_csv('data/X_train.csv', header=None).to_numpy()
Y_train = pd.read_csv('data/Y_train.csv', header=None).to_numpy()

print(f"X_train: {X_train.shape}")
print(f"Y_train: {Y_train.shape}")

####################################################################
# Transpose Data
X_train = X_train.T
Y_train = Y_train.T

####################################################################
# Normalize X values
X_train = X_train / 255.

####################################################################
nn = NeuralNetwork()

nn.add_layer(Layer(784, 200, Sigmoid()))
nn.add_layer(Layer(200, 200, Sigmoid()))
nn.add_layer(Layer(200, 10, Softmax()))

print("Training...")
nn.train(X_train,
         Y_train,
         epochs=2,
         learning_rate=0.1,
         loss=CatCrossEntropy(),
         history_path="testdir/python_history.csv")

print("Saving Model...")
nn.save_config("testdir/python_config.csv")
nn.save_weights("testdir/python_weights.csv")
