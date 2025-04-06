from abaclass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


outer_folds = 5
inner_folds = 5
lambda_vals = [0.001,0.1,1,10,100]

outer_error = []
best_lambda = []

outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(outer_cv.split(x_train_mat)):
    x_train_outer, x_test_outer = x_train_mat[train_idx], x_train_mat[test_idx]
    y_train_outer, y_test_outer = y_train_mat[train_idx], y_train_mat[test_idx]

    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=1)

    mean_error = []

    for lam in lambda_vals:
        C = 1/lam
        fold_error = []

        for train_inner_idx, inner_val_idx in inner_cv.split(x_train_outer):
            x_train_inner, x_val_inner = x_train_outer[train_inner_idx], x_train_outer[inner_val_idx]
            y_train_inner, y_val_inner = y_train_outer[train_inner_idx], y_train_outer[inner_val_idx]

            model = LogisticRegression(C=C, penalty='l2', solver='liblinear', max_iter=1000)
            model.fit(x_train_inner, y_train_inner)

            y_pred = model.predict(x_val_inner)
            fold_error.append(1 - accuracy_score(y_val_inner, y_pred))

        mean_error.append(np.mean(fold_error))

    best_lambda_idx = np.argmin(mean_error)
    best_lambda = lambda_vals[best_lambda_idx]
    best_C = 1/lambda_vals[best_lambda_idx]
    
    final_model = LogisticRegression(C=best_C, penalty='l2', solver='liblinear', max_iter=1000)
    final_model.fit(x_train_outer, y_train_outer)
    y_pred_outer = final_model.predict(x_test_outer)
    outer_error.append(1 - accuracy_score(y_test_outer, y_pred_outer))

    print(f"fold{i+1}: Best lambda = {best_lambda}, Error = {outer_error}")

print(f"Mean outer error: {np.mean(outer_error):.3f}")
print(f"Best lambda: {best_lambda}")
print(f"std: {np.std(outer_error):.3f}")