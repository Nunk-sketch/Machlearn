from abaclass import *
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# NOTE: Naive Bayes is only use for classification, not regression
# Define parameter grids for each model
param_grids = {
    "ANN": [0.001, 0.01, 0.1],           # λ = alpha
    "CT": [2, 4, 6, 8],                  # max_depth
    "KNN": [1, 3, 5, 7]                  # k = n_neighbors
}

# Storage
outer_folds = 5
inner_folds = 5
models = ["ANN", "CT", "KNN"]
results = {model: [] for model in models}
best_params = {model: [] for model in models}

outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

for model_name in models:
    print(f"\n=== Model: {model_name} ===")
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_train_mat_reg)):
        x_train_outer, x_test_outer = x_train_mat_reg[train_idx], x_train_mat_reg[test_idx]
        y_train_outer, y_test_outer = y_train_mat_reg[train_idx], y_train_mat_reg[test_idx]

        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=1)
        param_ev = []

        for param in param_grids[model_name]:
            fold_ev = []
            for inner_train_idx, val_idx in inner_cv.split(x_train_outer):
                x_train_inner, X_val = x_train_outer[inner_train_idx], x_train_outer[val_idx]
                y_train_inner, y_val = y_train_outer[inner_train_idx], y_train_outer[val_idx]

                # Train model based on type and param
                if model_name == "ANN":
                    model = MLPRegressor(hidden_layer_sizes=(100,), alpha=param, max_iter=1000)
                elif model_name == "CT":
                    model = DecisionTreeRegressor(max_depth=param)
                elif model_name == "KNN":
                    model = KNeighborsRegressor(n_neighbors=param)

                model.fit(x_train_inner, y_train_inner)
                y_pred_val = model.predict(X_val)
                fold_ev.append(explained_variance_score(y_val, y_pred_val))

            param_ev.append(np.mean(fold_ev))

        # Select best param from inner CV
        best_idx = np.argmin(param_ev)
        best_param = param_grids[model_name][best_idx]
        best_params[model_name].append(best_param)

        # Retrain on full outer train set with best param
        if model_name == "ANN":
            final_model = MLPRegressor(hidden_layer_sizes=(100,), alpha=best_param, max_iter=1000)
        elif model_name == "CT":
            final_model = DecisionTreeRegressor(max_depth=best_param)
        elif model_name == "KNN":
            final_model = KNeighborsRegressor(n_neighbors=best_param)

        final_model.fit(x_train_outer, y_train_outer)
        y_pred_test = final_model.predict(x_test_outer)
        explained_variance = explained_variance_score(y_test_outer, y_pred_test) # (ev)
        results[model_name].append(explained_variance)

        print(f"Fold {fold_idx+1}: Best param = {best_param}, Explained variance = {explained_variance:.3f}")


for model in models:
    print(f"\n--- {model} Summary ---")
    print(f"Best explained variance: {np.max(results[model]):.3f}")
    print(f"Mean explained variance: {np.mean(results[model]):.3f}")
    print(f"Std. explained variance: {np.std(results[model]):.3f}")
    print(f"Best params: {best_params[model]}")

print(f"the best model is: {max(results, key=lambda k: np.mean(results[k]))}")