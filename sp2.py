import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading breast cancer dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Creating classifier
classifier = GaussianNB()

# Define the hyperparameter search space
param_dist = {'var_smoothing': np.logspace(0, -9, num=100)}

# Performing random serarch
rand_search = RandomizedSearchCV(classifier, param_distributions=param_dist, cv=5, n_iter=100, n_jobs=-1, random_state=42)
rand_search.fit(X_train, y_train)
best_params = rand_search.best_params_

# Training the classifier with optimal hyperparameters
classifier = GaussianNB(var_smoothing=best_params['var_smoothing'])
classifier.fit(X_train, y_train)
# Evaluating performance
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print("Cross-validation accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Test set sensitivity:", sensitivity)
print("Test set specificity:", specificity)
