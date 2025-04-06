from abaclass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import Counter

most_common_class = Counter(y_train).most_common(1)[0][0]  

y_pred = np.full_like(y_test, fill_value= most_common_class) 
error_rate = np.mean(y_pred != y_test) 

# the error rate i 0.64, which is high, but this is the base case
# which we will use to compare with the other models

