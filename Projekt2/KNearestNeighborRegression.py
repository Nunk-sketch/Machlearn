from abaclass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

model_reg = KNeighborsRegressor(n_neighbors=10)  # Create a KNN model with k=10
model_reg.fit(x_train_reg, y_train_reg)  # Fit the model to training data
y_pred_reg = model_reg.predict(x_test_reg)  # Predict the output for test data
explained_variance = explained_variance_score(y_test_reg, y_pred_reg)
# model_reg.score(x_test_reg, y_pred_reg)  # Calculate R^2 score (same value if no bias)
print(f"Regression score (R2): {explained_variance:.2f}")  # Print error rate