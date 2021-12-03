"""
Małgorzata Cichowlas s16512
Before use: pip install sklearn
The goal of this program is to teach the dataset to predict accuracy of breast cancer.
Inputs are features like mean and worst dimensions of breast mass. 
Outputs are two diagnosis: maligant and benign mass (//guz złośliwy i łagodny).
Dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
Based on tutorial from website https://www.datacamp.com/community/tutorials/ and code from NAI lecture.  
"""

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Load and check dataset
cancer = datasets.load_breast_cancer()
print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)
print(cancer.data.shape)
#print(cancer.data[0:5])
#print(cancer.target)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=100) 

# Create a svm Classifier
k_linear = svm.SVC(kernel='linear', C=100) 
k_rbf = svm.SVC(kernel='rbf', C=1,gamma='auto') 
k_poly = svm.SVC(kernel='poly',C=1, degree=2,gamma='auto')

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




