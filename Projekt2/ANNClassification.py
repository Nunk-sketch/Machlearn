from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from abaclass import *

alphas = np.linspace(0.001, 1, num=8) # Regularization parameters

# Split D into training and test sets (0.8, 0.2)
x_train, x_test, y_train, y_test = train_test_split(x_train_mat_clas, y_train_mat_clas, test_size=0.2, random_state=42)

for alpha in alphas:
    model_ann = MLPClassifier(hidden_layer_sizes=100,alpha = alpha, max_iter = 5000)
    model_ann.fit(x_train, y_train)

    ypred_ann = model_ann.predict(x_test)
    error_rate_ann = 1 - accuracy_score(y_test, ypred_ann)
    accuracy = accuracy_score(y_test, ypred_ann)
    score = model_ann.score(x_test, ypred_ann)

    print(f"Accuracy: {accuracy:.3f}") # Print accuracy
    print(f"Error rate: {error_rate_ann:.3f}") # Print error rate
