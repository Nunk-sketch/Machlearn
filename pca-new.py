from data import *
from sklearn.preprocessing import StandardScaler

D_np = D_clean.to_numpy() # convert to numpy arrays
#D_std = D_np - np.ones((N, 1)) * D_np.mean(axis=0) # mean = 0
D_std = StandardScaler().fit_transform(D_np) # mean = 0, std = 1 (standardized)

corr_mat = np.corrcoef(D_std.T) # correlation matrix

eigenvalues, eigenvectors = np.linalg.eig(corr_mat) # eigenvalues and eigenvectors

#eigenvalue and eigenvector pairs
pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
pairs.sort(key = lambda x: x[0], reverse = True)

sorted_eigenval = [] # sorted eigenvalues
for i in pairs:
    sorted_eigenval.append(i[0])

total = sum(eigenvalues) # sum of eigenvalues
variance_explained = [(i/total)*100 for i in sorted_eigenval] # variance explained by each component (rho)
print(f"Variance explained: {variance_explained}")
cumulative_variance = np.cumsum(variance_explained) # cumulative variance explained
print(f"Cumulative variance: {cumulative_variance}")

threshold = 90

#Plot variance explained by the principal components
plt.figure(figsize=(8, 6))
num_components = len(variance_explained)
plt.plot(range(1, len(variance_explained) + 1), variance_explained, "x-") # , alpha=0.7, align='center', label='individual explained variance'
plt.plot(range(1, len(variance_explained) + 1), cumulative_variance, "o-") # , where='mid', label='cumulative explained variance'
plt.plot([1, len(variance_explained)], [threshold, threshold], "k--")
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.tight_layout()
plt.grid()
plt.savefig('variance_explained.png')
plt.show()

# Plot classifications
# Use 2 components to explain more than 90% of the variance (could use 3 for 95%)
projection_mat = np.hstack((pairs[0][1].reshape(8,1),
                           pairs[1][1].reshape(8,1)))

D_new = D_std.dot(projection_mat) # project the data onto the principal components

# Plot the PCA with rings
plt.figure(figsize=(8, 6))
scatter = plt.scatter(D_new[:, 0], D_new[:, 1], c=D['Rings'], cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Abalone dataset')
plt.colorbar(scatter, label='Number of Rings')
plt.grid()
plt.savefig('pca-graph.png')
plt.show()