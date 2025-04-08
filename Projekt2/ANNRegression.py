from sklearn.neural_network import MLPRegressor

from abaclass import *

regressor = MLPRegressor(hidden_layer_sizes=(200), alpha=0.001, max_iter=5000)
regressor.fit(x_train_reg, y_train_reg) # Train on training data
score = regressor.score(x_test_reg, y_test_reg) # Test on test data

print(f"Regression score (R2): {score:.2f}") # Print R^2 score