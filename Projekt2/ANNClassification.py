from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from abaclass import *

model_ann = MLPClassifier(hidden_layer_sizes=(10),alpha = 0.01, max_iter = 5000)
model_ann.fit(x_train_clas, y_train_clas)

ypred_ann = model_ann.predict(x_test_clas)
error_rate_ann = 1 - accuracy_score(y_test_clas, ypred_ann)
print(f"Error rate: {error_rate_ann:.2f}") # Print error rate 13
