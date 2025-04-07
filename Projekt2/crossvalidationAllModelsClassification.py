from abaclass import *
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# Define parameter grids for models
param_grids = {
    "ANN": [0.001, 0.01, 0.1],           # λ = alpha
    "CT": [2, 4, 6, 8],                  # max_depth
    "KNN": [1, 3, 5, 7],                 # k = n_neighbors
    "NB": [0.1, 0.5, 1.0, 2.0]           # b = alpha
}

# Storage
outer_folds = 5
inner_folds = 5
models = ["ANN", "CT", "KNN", "NB"]
results = {model: [] for model in models}
best_params = {model: [] for model in models}

outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

for model_name in models:
    print(f"\n=== Model: {model_name} ===")
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_train_mat_clas)):
        x_train_outer, x_test_outer = x_train_mat_clas[train_idx], x_train_mat_clas[test_idx]
        y_train_outer, y_test_outer = y_train_mat_clas[train_idx], y_train_mat_clas[test_idx]

        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=1)
        param_errors = []

        for param in param_grids[model_name]:
            fold_errors = []
            for inner_train_idx, val_idx in inner_cv.split(x_train_outer):
                x_train_inner, X_val = x_train_outer[inner_train_idx], x_train_outer[val_idx]
                y_train_inner, y_val = y_train_outer[inner_train_idx], y_train_outer[val_idx]

                # Train model based on type and param
                if model_name == "ANN":
                    model = MLPClassifier(hidden_layer_sizes=(100,), alpha=param, max_iter=1000)
                elif model_name == "CT":
                    model = DecisionTreeClassifier(max_depth=param)
                elif model_name == "KNN":
                    model = KNeighborsClassifier(n_neighbors=param)
                elif model_name == "NB":
                    model = MultinomialNB(alpha=param)

                model.fit(x_train_inner, y_train_inner)
                y_pred_val = model.predict(X_val)
                fold_errors.append(1 - accuracy_score(y_val, y_pred_val))

            param_errors.append(np.mean(fold_errors))

        # Select best param from inner CV
        best_idx = np.argmin(param_errors)
        best_param = param_grids[model_name][best_idx]
        best_params[model_name].append(best_param)

        # Retrain on full outer train set with best param
        if model_name == "ANN":
            final_model = MLPClassifier(hidden_layer_sizes=(100,), alpha=best_param, max_iter=1000)
        elif model_name == "CT":
            final_model = DecisionTreeClassifier(max_depth=best_param)
        elif model_name == "KNN":
            final_model = KNeighborsClassifier(n_neighbors=best_param)
        elif model_name == "NB":
            final_model = MultinomialNB(alpha=best_param)

        final_model.fit(x_train_outer, y_train_outer)
        y_pred_test = final_model.predict(x_test_outer)
        error = 1 - accuracy_score(y_test_outer, y_pred_test)
        results[model_name].append(error)

        print(f"Fold {fold_idx+1}: Best param = {best_param}, Error = {error:.3f}")


for model in models:
    print(f"\n--- {model} Summary ---")
    print(f"Mean error: {np.mean(results[model]):.3f}")
    print(f"Std. error: {np.std(results[model]):.3f}")
    print(f"Best params: {best_params[model]}")

print(f"the best model is: {min(results, key=lambda k: np.mean(results[k]))}")