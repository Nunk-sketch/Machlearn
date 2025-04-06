from data import *
from sklearn.decomposition import PCA
from scipy import stats

# Standardize the data
D_std = (D_clean - D_clean.mean()) / D_clean.std()

X = D_std.drop(columns=["Rings"])
Y = D_std
X_and_Y = pd.concat([X, Y], axis=1)

pca = PCA()
pca.fit(X)
X_pca = pca.transform(X)

#%% Remove outliers
# Calculate Z-scores
z_scores = np.abs(stats.zscore(X_pca))

# Define a threshold for Z-scores
threshold = 5

# Filter out the outliers
outlier_mask = (z_scores < threshold).all(axis=1)
X_pca = X_pca[outlier_mask]
X_and_Y = X_and_Y[outlier_mask]

print(f"Original shape: {(4177, 7)}")
print(f"Filtered shape: {X_pca.shape}")
print(f"Number of outliers: {4177 - X_pca.shape[0]}")

#%% Plot the explained variance (this section could also use the PCA package, it was tested with and without and gave same results)
corr_mat = np.corrcoef(X.T) # correlation matrix

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

threshold = 92.5

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
plt.savefig('billeder/variance_explained.png')
plt.show()

#%% Plot the principal components
i, j = 3, 4 # components to plot

# Plot the PCA with rings
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, i], X_pca[:, j], c=X_and_Y['Rings'], cmap='viridis', s=100) # edgecolor='k'
plt.xlabel('Principal Component {}'.format(i+1))
plt.ylabel('Principal Component {}'.format(j+1))
plt.title('PCA of Abalone dataset')
plt.colorbar(scatter, label='Number of Rings')
plt.grid()

# Plot attribute coefficients in principal component space
n_attributes = 7
for att in range(n_attributes):
    plt.arrow(0, 0, X_pca[att, i], X_pca[att, j], color='0', head_width=0.05, head_length=0.1, linewidth=2)
    plt.text(X_pca[att, i], X_pca[att, j], parameters[att], color='r', fontsize=12)

# Save figure and show the plot
plt.savefig('billeder/pca-graph.png')
plt.show()