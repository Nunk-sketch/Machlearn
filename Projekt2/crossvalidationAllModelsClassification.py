from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from networkModels import FCNN
import numpy as np
from abaclass import *


# Define parameter grids for models
param_grids = {
    "ANN": {"alpha": [0.001, 0.01, 0.1, 1, 2, 4]},
    "CT": {"max_depth": [2, 4, 6, 8, 10, 12]},
    "KNN": {"n_neighbors": [1, 3, 5, 7, 9, 11]},
    "NB": {"alpha": [0.1, 0.5, 1.0, 2.0, 3, 4]},
    "MN": {"C": [1 / val for val in [0.001, 0.01, 0.1, 1, 2, 4]]},
    "FCNN": {"epochs": [10, 20, 40, 80, 160, 500]},
}

# Storage
outer_folds = 5
models = ["ANN", "CT", "KNN", "NB", "MN", "FCNN"]
results = {model: [] for model in models}
best_params = {model: [] for model in models}

outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

# Helper function to initialize models
def initialize_model(model_name, params):
    if model_name == "ANN":
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, **params)
    elif model_name == "CT":
        return DecisionTreeClassifier(**params)
    elif model_name == "KNN":
        return KNeighborsClassifier(**params)
    elif model_name == "NB":
        return MultinomialNB(**params)
    elif model_name == "MN":
        return LogisticRegression(solver='lbfgs', max_iter=1000, **params)
    elif model_name == "FCNN":
        return FCNN()

# Outer cross-validation
for model_name in models:
    print(f"\n=== Model: {model_name} ===")
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_train_mat_clas)):
        x_train_outer, x_test_outer = x_train_mat_clas[train_idx], x_train_mat_clas[test_idx]
        y_train_outer, y_test_outer = y_train_mat_clas[train_idx], y_train_mat_clas[test_idx]

        if model_name == "FCNN":
            # Manual inner CV for FCNN
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
            param_errors = []
            for param in param_grids[model_name]["epochs"]:
                fold_errors = []
                for inner_train_idx, val_idx in inner_cv.split(x_train_outer):
                    x_train_inner, X_val = x_train_outer[inner_train_idx], x_train_outer[val_idx]
                    y_train_inner, y_val = y_train_outer[inner_train_idx], y_train_outer[val_idx]

                    model = FCNN()
                    best_val_loss = float('inf')
                    patience = 5
                    patience_counter = 0

                    for epoch in range(param):
                        model.train_model(x_train_inner, y_train_inner, X_val, y_val, epochs=1)
                        val_loss = model.compute_loss(X_val, y_val)  # Replace with the correct method or implement evaluate_loss in FCNN

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            break

                    y_pred_val = model.predict(X_val)
                    fold_errors.append(1 - accuracy_score(y_val, y_pred_val))

                param_errors.append(np.mean(fold_errors))

            best_idx = np.argmin(param_errors)
            best_param = param_grids[model_name]["epochs"][best_idx]
            best_params[model_name].append(best_param)

            model = FCNN()
            model.train_model(x_train_outer, y_train_outer, x_test_outer, y_test_outer, epochs=best_param)
            y_pred_test = model.predict(x_test_outer)
            error = 1 - accuracy_score(y_test_outer, y_pred_test)
            results[model_name].append(error)

        else:
            # Use GridSearchCV for other models
            model = initialize_model(model_name, {})
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(x_train_outer, y_train_outer)

            best_params[model_name].append(grid_search.best_params_)
            best_model = grid_search.best_estimator_
            y_pred_test = best_model.predict(x_test_outer)
            error = 1 - accuracy_score(y_test_outer, y_pred_test)
            results[model_name].append(error)

        print(f"Fold {fold_idx+1}: Best param = {best_params[model_name][-1]}, Error = {error:.3f}")

# Summary
for model in models:
    print(f"\n--- {model} Summary ---")
    print(f"Mean error: {np.mean(results[model]):.3f}")
    print(f"Std. error: {np.std(results[model]):.3f}")
    print(f"Best params: {best_params[model]}")

print(f"The best model is: {min(results, key=lambda k: np.mean(results[k]))}")
