import time

from dtuimldmtools import rlr_validate
from skimage.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from abaclass import *
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Define parameter grids for each model
param_grids = {
    "ANN": [0.001, 0.01, 0.1],           # λ = alpha
    "RLR": [1, 0.1, 0.01, 0.001],        # λ = generalization factor
    "BASE": [0, 0, 0, 0]                 # param not used
}

# Storage
outer_folds = 10
inner_folds = 10
models = ["ANN", "RLR", "BASE"]
results = {model: [] for model in models}
best_params = {model: [] for model in models}

skip_models = [] #["ANN", "RLR"]

outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

for model_name in models:
    print(f"\n=== Model: {model_name} ===")
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_train_mat_reg)):
        x_train_outer, x_test_outer = x_train_mat_reg[train_idx], x_train_mat_reg[test_idx]
        y_train_outer, y_test_outer = y_train_mat_reg[train_idx], y_train_mat_reg[test_idx]

        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=1)
        param_mse = []

        if model_name == "RLR":
            lambdas = param_grids[model_name]

        for param in param_grids[model_name]:
            fold_mse = []
            for inner_train_idx, val_idx in inner_cv.split(x_train_outer):
                x_train_inner, x_test_inner = x_train_outer[inner_train_idx], x_train_outer[val_idx]
                y_train_inner, y_test_inner = y_train_outer[inner_train_idx], y_train_outer[val_idx]

                # Train model based on type and param
                if model_name == "ANN":
                    model = MLPRegressor(hidden_layer_sizes=(100,), alpha=param, max_iter=1000)
                elif model_name == "RLR":
                    (
                        opt_val_err,
                        opt_lambda,
                        mean_w_vs_lambda,
                        train_err_vs_lambda,
                        test_err_vs_lambda,
                    ) = rlr_validate(x_train_inner, y_train_inner, lambdas, inner_folds)
                    model = Ridge(alpha=opt_lambda, fit_intercept=True)
                elif model_name == "BASE":
                    # Dummy model for baseline
                    # Calculate mean of training set
                    y_mean = np.mean(y_train_inner)

                    # Predict mean for validation set
                    y_pred_test = np.full_like(y_test_inner, y_mean)

                    # Calculate mean squared error
                    mse = mean_squared_error(y_test_inner, y_pred_test)

                    # Calculate explained varianced (R^2)
                    #explained_variance = explained_variance_score(y_test_inner, y_pred_test)

                    # Append the explained variance to fold_ev
                    fold_mse.append(mse)
                    continue

                model.fit(x_train_inner, y_train_inner)
                y_pred_val = model.predict(x_test_inner)
                fold_mse.append(mean_squared_error(y_test_inner, y_pred_val))

            param_mse.append(np.mean(fold_mse))

        # Select best param from inner CV
        best_idx = np.argmin(param_mse)
        best_param = param_grids[model_name][best_idx]
        best_params[model_name].append(best_param)

        # Retrain on full outer train set with best param
        if model_name == "ANN":
            final_model = MLPRegressor(hidden_layer_sizes=(100,), alpha=best_param, max_iter=1000)
        elif model_name == "RLR":
            final_model = Ridge(alpha=best_param, fit_intercept=True)
        elif model_name == "BASE":
            # Dummy model for baseline
            y_mean = np.mean(y_train_outer)
            y_pred_test = np.full_like(y_test_outer, y_mean)

            mse = mean_squared_error(y_test_outer, y_pred_test)

            results[model_name].append(mse)

            print(f"Fold {fold_idx+1}: Mean squared error = {mse:.3f}") #Explained variance = {explained_variance:.3f}")
            continue

        final_model.fit(x_train_outer, y_train_outer)
        y_pred_test = final_model.predict(x_test_outer)
        mse = mean_squared_error(y_test_outer, y_pred_test)
        #explained_variance = explained_variance_score(y_test_outer, y_pred_test) # (ev)
        results[model_name].append(mse)

        print(f"Fold {fold_idx+1}: Best param = {best_param}, Mean squared error = {mse:.3f}")
    print(f"Model run-time: {time.time() - start_time:.2f} seconds")


for model in models:
    if not results[model]:
        continue # Skip if no results
    print(f"\n--- {model} Summary ---")
    print(f"Lowest error: {np.min(results[model]):.3f}")
    print(f"Mean error: {np.mean(results[model]):.3f}")
    print(f"Std. error: {np.std(results[model]):.3f}")
    print(f"Best params: {best_params[model]}")

print(f"the best model is: {min(results, key=lambda k: np.mean(results[k]))}")