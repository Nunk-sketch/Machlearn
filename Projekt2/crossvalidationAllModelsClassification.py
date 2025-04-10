import json

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from abaclass import *

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
outer_folds = 10  # Increased the number of outer folds
models = ["ANN", "CT", "KNN", "NB", "MN", "BC"]
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
    elif model_name == "BC":
        return DummyClassifier(**params)

# Outer cross-validation
for model_name in models:
    print(f"\n=== Model: {model_name} ===")
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x_train_mat_clas)):
        print(f"Processing fold {fold_idx + 1}/{outer_folds} for model {model_name}...")
        
        x_train_outer, x_test_outer = x_train_mat_clas[train_idx], x_train_mat_clas[test_idx]
        y_train_outer, y_test_outer = y_train_mat_clas[train_idx], y_train_mat_clas[test_idx]

        # Use GridSearchCV for hyperparameter tuning
        model = initialize_model(model_name, {})
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
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
print(f"\nThe best model is: {best_model_name} with mean error {np.mean(results[best_model_name]):.3f}")

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

output_file = "crossvalidation_results.json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"\nResults saved to {output_file}")