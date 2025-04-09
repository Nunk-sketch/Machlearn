from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from abaclass import *
import json
from sklearn.dummy import DummyClassifier
from imblearn.under_sampling import RandomUnderSampler

# Define parameter grids for models
param_grids = {
    "ANN": {"alpha": [0.001, 0.01, 0.1, 1, 2, 4]},
    "CT": {"max_depth": [2, 4, 6, 8, 10, 12]},
    "KNN": {"n_neighbors": [1, 3, 5, 7, 9, 11]},
    "NB": {"alpha": [0.1, 0.5, 1.0, 2.0, 3, 4]},
    "MN": {"C": [1 / val for val in [0.001, 0.01, 0.1, 1, 2, 4]]},
    "BC": {"strategy": ["most_frequent"]}  # DummyClassifier for baseline
}

# Storage
outer_folds = 10
models = ["ANN", "CT", "KNN", "NB", "MN", "BC"]
results = {model: [] for model in models}
best_params = {model: [] for model in models}

outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)


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
    elif model_name == "BC":
        return DummyClassifier(**params)

# Outer cross-validation
for model_name in models:
    print(f"\n=== Model: {model_name} ===")
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_clas_mat)):
        print(f"Processing fold {fold_idx + 1}/{outer_folds} for model {model_name}...")
        
        x_train_outer, x_test_outer = x_clas_mat[train_idx], x_clas_mat[test_idx]
        y_train_outer, y_test_outer = y_clas_mat[train_idx], y_clas_mat[test_idx]
        min_class_count = min(np.bincount(y_train_outer))
        undersampler = RandomUnderSampler(
            sampling_strategy={0: min(300, min_class_count), 1: min(300, min_class_count), 2: min(300, min_class_count)},  # Adjusted to avoid exceeding class counts
            random_state=42
        )
        x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train_outer, y_train_outer)


        # Use GridSearchCV for hyperparameter tuning
        model = initialize_model(model_name, {})
        grid_search = GridSearchCV(model, param_grids[model_name], cv=10, scoring='accuracy', n_jobs=-1) # inner folds: 
        grid_search.fit(x_train_outer, y_train_outer)

        best_params[model_name].append(grid_search.best_params_)
        best_model = grid_search.best_estimator_
        y_pred_test = best_model.predict(x_test_outer)
        error = 1 - accuracy_score(y_test_outer, y_pred_test)
        results[model_name].append(error)

        print(f"Fold {fold_idx + 1}: Best param = {best_params[model_name][-1]}, Error = {error:.3f}")

# Summary
print("\n=== Summary ===")
for model in models:
    print(f"\n--- {model} ---")
    print(f"Mean error: {np.mean(results[model]):.3f}")
    print(f"Std. error: {np.std(results[model]):.3f}")
    print(f"Best params: {best_params[model]}")

best_model_name = min(results, key=lambda k: np.mean(results[k]))
print(f"\nThe best model is: {best_model_name} with mean error {np.mean(results[best_model_name]):.4f}")

# Save results to a JSON file
output_data = {
    "results": {model: results[model] for model in models},
    "best_params": {model: best_params[model] for model in models},
    "best_model": {
        "name": best_model_name,
        "mean_error": np.mean(results[best_model_name]),
        "std_error": np.std(results[best_model_name]),
    },
}
# Determine the overall best parameters for each model
overall_best_params = {}
for model in models:
    param_counts = {}
    for params in best_params[model]:
        param_tuple = tuple(params.items())
        param_counts[param_tuple] = param_counts.get(param_tuple, 0) + 1
    overall_best_params[model] = max(param_counts, key=param_counts.get)

# Convert tuples back to dictionaries for readability
overall_best_params = {model: dict(params) for model, params in overall_best_params.items()}

# Add overall best parameters to the output data
output_data["overall_best_params"] = overall_best_params

output_file = "crossvalidation_resultsNY2.json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"\nResults saved to {output_file}")