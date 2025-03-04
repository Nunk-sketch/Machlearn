from data import *
import matplotlib.pyplot as plt
from scipy.linalg import svd

D_np = D_clean.to_numpy()
Y = D_np - np.ones((N, 1)) * D_np.mean(axis=0)

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
plt.show()

# Generate PCA plot
Z = Y @ V # matrix multiplication

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title("Abalone: PCA")
# Z = array(Z)

for param in parameters:
    # select indices belonging to class c:
    class_mask = D_clean[param] == D_clean[param].unique()
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)

plt.legend(parameters)
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))

# Output result to screen
# plt.save("PCA_plot_new.png")
plt.show()