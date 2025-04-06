from data import *

# Compare the statistics of male, female, and infant abalones
comparison = pd.DataFrame({
    "Male": maleinfo.mean(),
    "Female": femaleinfo.mean(),
    "Infant": infantinfo.mean()
})

# Standardize the datasets
mean = np.mean(D, axis=0)
std = np.std(D, axis=0)
D_standardized = (D - mean) / std

mean_male = np.mean(D_male, axis=0)
std_male = np.std(D_male, axis=0)
D_male_standardized = (D_male - mean_male) / std_male

mean_female = np.mean(D_female, axis=0)
std_female = np.std(D_female, axis=0)
D_female_standardized = (D_female - mean_female) / std_female

mean_infant = np.mean(D_infant, axis=0)
std_infant = np.std(D_infant, axis=0)
D_infant_standardized = (D_infant - mean_infant) / std_infant

# Calculate covariance matrices
cov = np.cov(D_standardized, rowvar=False)
cov_male = np.cov(D_male_standardized, rowvar=False)
cov_female = np.cov(D_female_standardized, rowvar=False)
cov_infant = np.cov(D_infant_standardized, rowvar=False)


print("Comparison of mean values:\n", comparison)

#male correlation
correlation_matrix = D_male_standardized.corr()

# Print the correlation matrix
print("Correlation matrix for male abalones:\n", correlation_matrix)

#female correlation
correlation_matrix = D_female_standardized.corr()

# Print the correlation matrix
print("Correlation matrix for female abalones:\n", correlation_matrix)

#infant correlation
correlation_matrix = D_infant_standardized.corr()

# Print the correlation matrix
print("Correlation matrix for infant abalones:\n", correlation_matrix)

# Total correlation
correlation_matrix = D_standardized.corr()

# Print the correlation matrix
print("Correlation matrix for all abalones:\n", correlation_matrix)


# Male covariance
covariance_matrix_male = D_male_standardized.cov()

# Print the covariance matrix
print("Covariance matrix for male abalones:\n", covariance_matrix_male)

# Female covariance
covariance_matrix_female = D_female_standardized.cov()

# Print the covariance matrix
print("Covariance matrix for female abalones:\n", covariance_matrix_female)

# Infant covariance
covariance_matrix_infant = D_infant_standardized.cov()

# Print the covariance matrix
print("Covariance matrix for infant abalones:\n", covariance_matrix_infant)

# Total covariance
covariance_matrix = D_standardized.cov()

# Print the covariance matrix
print("Covariance matrix for all abalones:\n", covariance_matrix)