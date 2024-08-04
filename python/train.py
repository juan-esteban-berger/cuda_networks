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
# X_train = pd.read_csv('data/X_train.csv', header=None).head(10000).to_numpy()
# Y_train = pd.read_csv('data/Y_train.csv', header=None).head(10000).to_numpy()
# 
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

batch_size = 8
nn.add_layer(Layer(784, 200, Sigmoid(), batch_size))
nn.add_layer(Layer(200, 200, Sigmoid(), batch_size))
nn.add_layer(Layer(200, 10, Softmax(), batch_size))

print("Training...")
nn.train(X_train,
         Y_train,
         # epochs=200,
         epochs=3,
         learning_rate=0.1,
         loss=CatCrossEntropy(),
         optimizer="mini_batch_gradient_descent",
         batch_size=batch_size,
         history_path="models/python_history.csv")

print("Saving Model...")
nn.save_config("models/python_config.csv")
nn.save_weights("models/python_weights.csv")
