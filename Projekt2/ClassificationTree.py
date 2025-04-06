from abaclass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#classification tree
model_ct = DecisionTreeClassifier(max_depth = 4)
model_ct.fit(x_train,y_train)

y_pred_ct = model_ct.predict(x_test)
error_rate_ct = 1 - accuracy_score(y_test, y_pred_ct)
print(f"Error rate: {error_rate_ct:.2f}") # Print error rate