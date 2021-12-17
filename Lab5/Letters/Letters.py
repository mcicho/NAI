"""
Małgorzata Cichowlas s16512
Before use: pip install numpy, pip install sklearn
The goal of this program is to teach the dataset to recognize and predict accuracy of letters based on their sizes, edges and etc.
Dataset: https://www.openml.org/d/6 , https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
Based on tutorial from websites https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html , 
http://athena.ecs.csus.edu/ and code from NAI lecture.
"""

import warnings
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# Load data from https://www.openml.org/d/6
X, y = fetch_openml("letter", version = 1,return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=20) 


# Create a MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100),
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
print("Test set score: %f" % mlp.score(X_test, y_test))
