from collections import Counter

from abaclass import *

most_common_class = Counter(y_train_clas).most_common(1)[0][0]

y_pred = np.full_like(y_test_clas, fill_value= most_common_class)  # Fill y_pred with the most common class

error_rate = np.mean(y_pred != y_test_clas)  # Calculate the error rate

print(f"Error rate: {error_rate:.2f}")  # Print error rate
print(f"Most common class: {most_common_class}")  # Print most common class
