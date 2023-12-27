import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd

X_train = pd.read_csv('X_train.csv', header=None)
X_train

Y_train = pd.read_csv('Y_train.csv', header=None)
Y_train

X_test = pd.read_csv('X_test.csv', header=None)
X_test

Y_test = pd.read_csv('Y_test.csv', header=None)
Y_test
