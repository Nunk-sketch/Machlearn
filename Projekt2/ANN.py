from abaclass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


model_ann = MLPClassifier(hidden_layer_sizes=(200),alpha = 0.001, max_iter = 5000)
model_ann.fit(x_train, y_train)

ypred_ann = model_ann.predict(x_test)
error_rate_ann = 1 - accuracy_score(y_test, ypred_ann)
print(f"Error rate: {error_rate_ann:.2f}") # Print error rate
