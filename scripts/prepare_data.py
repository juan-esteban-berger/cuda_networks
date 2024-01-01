# Setup
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import OneHotEncoder

# Import Data
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Split into feature and target datasets
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Plot the first image
plt.imshow(train_df.iloc[0, 1:].values.reshape(28, 28), cmap='gray')
plt.show()

# Preview Features
X_train.head()

# Preview Targets
pd.DataFrame(y_train).head()

# initialize One-Hot Encoder
one_hot = OneHotEncoder()

# One-Hot encode the targets
Y_train = one_hot.fit_transform(y_train.values.reshape(-1,1)).toarray()
Y_test = one_hot.transform(y_test.values.reshape(-1,1)).toarray()

# Preview One-Hot Encoded Targets
Y_train[0:5]

# Save Datasets to csv without column headers
X_train.to_csv('X_train.csv', header=False, index=False)
X_test.to_csv('X_test.csv', header=False, index=False)
pd.DataFrame(Y_train).to_csv('Y_train.csv', header=False, index=False)
pd.DataFrame(Y_test).to_csv('Y_test.csv', header=False, index=False)
