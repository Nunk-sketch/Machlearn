from skimage.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from abaclass import *

alphas = np.linspace(0.001, 1, num=8) # Regularization parameters

for alpha in alphas:
    regressor = MLPRegressor(hidden_layer_sizes=100, alpha=0.001, max_iter=5000)
    regressor.fit(x_train_reg, y_train_reg) # Train on training data
    score = regressor.score(x_test_reg, y_test_reg) # Test on test data
    mse = mean_squared_error(y_test_reg, regressor.predict(x_test_reg)) # Calculate mean squared error

    print(f"Regression score (R2): {score:.2f} - Alpha: {alpha}") # Print R^2 score
    print(f"Mean Squared Error: {mse:.2f} - Alpha: {alpha}") # Print mean squared error
