from data import *
import matplotlib.pyplot as plt
from scipy.linalg import svd

Y = D_clean - D_clean.mean(axis=0) # np.ones((N, 1)) * D_clean.mean(axis=0)

U, S, Vh = svd(Y, full_matrices=False)

V = Vh.T

rho = (S * S) / (S * S).sum()

threshold = 0.95

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.savefig('variance_explained.png')
plt.show()