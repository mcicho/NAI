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
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Load and check dataset
data = 'C:/Users/48667/Documents/Python Scripts/04_src/Lab4/Seeds/seeds.csv'
seeds = pd.read_csv(data)
print(seeds.shape)
#print(seeds.head())
col_names = seeds.columns
print(col_names)
#print(seeds['wheat'].value_counts())
#print(seeds['wheat'].value_counts()/np.float(len(seeds)))
#print(seeds.isnull().sum())

# Define X, y from dataset
X = seeds.drop(['wheat'], axis=1)
y = seeds['wheat']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=20) 

# Create a svm Classifier
k_linear = svm.SVC(kernel='linear') 
k_rbf = svm.SVC(kernel='rbf') 
k_poly = svm.SVC(kernel='poly', degree=3)

# Train the model using the training sets
k_linear.fit(X_train, y_train)
k_rbf.fit(X_train, y_train)
k_poly.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_linear = k_linear.predict(X_test)
y_pred_rbf = k_rbf.predict(X_test)
y_pred_poly= k_poly.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy for k_linear:",metrics.accuracy_score(y_test, y_pred_linear))
print("Accuracy for k_rbf:",metrics.accuracy_score(y_test, y_pred_rbf))
print("Accuracy for k_poly:",metrics.accuracy_score(y_test, y_pred_poly))