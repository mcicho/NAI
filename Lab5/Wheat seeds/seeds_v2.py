"""
Małgorzata Cichowlas s16512
Before use: pip install numpy , pip install pandas , pip install sklearn
The goal of this program is to teach the dataset to predict accuracy of three different varieties of wheat: Kama, Rosa and Canadian.
Inputs are features of kernel of seed. 
Outputs are three different varieties of wheat.
Dataset: https://archive.ics.uci.edu/ml/datasets/seeds
Based on tutorial from website https://www.datacamp.com/community/tutorials/ and code from NAI lecture.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt


# Load and check dataset
data = 'C:/Users/48667/Documents/Python Scripts/Lab5/seeds.csv'
seeds = pd.read_csv(data)
#print(seeds.shape)
#print(seeds.head())
#col_names = seeds.columns
#print(col_names)
#print(seeds['wheat'].value_counts())
#print(seeds['wheat'].value_counts()/np.float(len(seeds)))
#print(seeds.isnull().sum())

# Define X, y from dataset
X = seeds.drop(['wheat'], axis=1)
y = seeds['wheat']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=20) 

# Create a MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(50),
    max_iter=10,
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
)

# Handling sgd errors and train the model using the training sets
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)


print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set mlp.score: %f" % mlp.score(X_test, y_test))
