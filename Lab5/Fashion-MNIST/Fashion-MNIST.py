"""
Małgorzata Cichowlas s16512
Before use: pip install numpy, pip install sklearn, pip install matplotlib
The goal of this program is to teach the dataset to recognize and predict accuracy of ten kind of clothes from image-dataset.  
Dataset: https://www.openml.org/d/40996 , https://github.com/zalandoresearch/fashion-mnist
Based on tutorial from websites https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html , 
http://athena.ecs.csus.edu/ and code from NAI lecture.
"""

import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np


# Load data from https://www.openml.org/d/40996
X, y = fetch_openml("Fashion-MNIST", return_X_y=True)
X = X / 255.0
labelNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Rescale the data, use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Create a MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,200),
    max_iter=10,
    alpha=1e-4,
    solver="lbfgs",
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

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Confusion matrix
y_pred_mlp = mlp.predict(X_test)
mlp_f1 = metrics.f1_score(y_test, y_pred_mlp, average= "weighted")
mlp_accuracy = metrics.accuracy_score(y_test, y_pred_mlp)
mlp_cm = metrics.confusion_matrix(y_test, y_pred_mlp)

print("Confusion matrix: \n", mlp_cm)
print('Plotting confusion matrix')

plt.figure()
plot_confusion_matrix(mlp_cm, labelNames)
plt.show()