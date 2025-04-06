from abaclass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

model_nb = MultinomialNB(alpha=0.01)
model_nb.fit(x_train,y_train)

y_pred_np = model_nb.predict(x_test)

error_rate_nb = 1 - accuracy_score(y_test, y_pred_np)
print("Error rate for Naive Bayes: ", error_rate_nb)
