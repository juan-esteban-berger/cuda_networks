import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from neural_network import NeuralNetwork

####################################################################
# Load Data
print("Loading Data...")
X_test = pd.read_csv('data/X_test.csv', header=None).to_numpy()
Y_test = pd.read_csv('data/Y_test.csv', header=None).to_numpy()

print(f"X_test: {X_test.shape}")
print(f"Y_test: {Y_test.shape}")

####################################################################
# Transpose Data
X_test = X_test.T
Y_test = Y_test.T

####################################################################
# Normalize X values
X_test = X_test / 255.

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
