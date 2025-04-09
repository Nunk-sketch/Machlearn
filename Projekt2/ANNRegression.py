from sklearn.neural_network import MLPRegressor

from abaclass import *
from sklearn.model_selection import cross_val_score, KFold

regressor = MLPRegressor(hidden_layer_sizes=(200), alpha=0.001, max_iter=5000)
regressor.fit(x_train_reg, y_train_reg) # Train on training data
score = regressor.score(x_test_reg, y_test_reg) # Test on test data

print(f"Regression score (R2): {score:.2f}") # Print R^2 score
# Define outer cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform outer cross-validation
outer_scores = []
for train_idx, test_idx in outer_cv.split(x_train_reg):
    x_train_outer, x_test_outer = x_train_reg[train_idx], x_train_reg[test_idx]
    y_train_outer, y_test_outer = y_train_reg[train_idx], y_train_reg[test_idx]

    # Define inner cross-validation
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_scores = cross_val_score(regressor, x_train_outer, y_train_outer, cv=inner_cv)

    # Train on the outer training set and evaluate on the outer test set
    regressor.fit(x_train_outer, y_train_outer)
    outer_score = regressor.score(x_test_outer, y_test_outer)
    outer_scores.append(outer_score)

print(f"Outer cross-validation scores: {outer_scores}")
print(f"Mean outer cross-validation score: {sum(outer_scores) / len(outer_scores):.2f}")