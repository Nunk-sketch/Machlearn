from abaclass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

model_knn = KNeighborsClassifier(n_neighbors=10)  # Create a KNN model with k=10
model_knn.fit(x_train_clas, y_train_clas)  # Fit the model to training data
y_pred_knn = model_knn.predict(x_test_clas)  # Predict the output for test data
error_rate_knn = 1 - accuracy_score(y_test_clas, y_pred_knn)  # Calculate error rate
print(f"Classification error rate: {error_rate_knn:.2f}")  # Print error rate