from abaclass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

lambda_val = 0.001 #lambda value for regularization
C_val = 1/lambda_val # C value for regularization




model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=C_val) # Create a logistic regression model with L2 regularization
model.fit(x_train_clas, y_train_clas) # Fit the model to the training data

y_pred = model.predict(x_test_clas) # Predict the output for the test data

error_rate = 1 - accuracy_score(y_test_clas, y_pred) # Calculate the error rate
# error rate is aprox 0.44 which is a significant improvement over the base case(with lambda = 0.1)
accuracy = accuracy_score(y_test_clas, y_pred) # Calculate the accuracy
print(f"Accuracy: {accuracy:.3f}")
print(f"Error rate: {error_rate:.3f}")

