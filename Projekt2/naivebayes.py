from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from abaclass import *

model_nb = MultinomialNB(alpha=0.01)
model_nb.fit(x_train_clas, y_train_clas)

y_pred_np = model_nb.predict(x_test_clas)

error_rate_nb = 1 - accuracy_score(y_test_clas, y_pred_np)
print("Error rate for Naive Bayes: ", error_rate_nb)
