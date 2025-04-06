import matplotlib.pyplot as plt
import statsmodels.api as sm
from data import *

# Scatter plot of Rings vs Shell weight
plt.figure()
plt.scatter(D['Rings'], D['Shell weight'])
plt.title('Scatter plot of Rings vs Shell weight')
plt.xlabel('Rings')
plt.ylabel('Shell weight')
plt.show()
