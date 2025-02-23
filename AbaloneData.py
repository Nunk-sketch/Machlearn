from data import *

# Compare the statistics of male, female, and infant abalones
comparison = pd.DataFrame({
    "Male": maleinfo.mean(),
    "Female": femaleinfo.mean(),
    "Infant": infantinfo.mean()
})

print("Comparison of mean values:\n", comparison)

#male correlation
correlation_matrix = D_male.drop(columns=['Sex']).corr()

# Print the correlation matrix
print("Correlation matrix for male abalones:\n", correlation_matrix)

#female correlation
correlation_matrix = D_female.drop(columns=['Sex']).corr()

# Print the correlation matrix
print("Correlation matrix for female abalones:\n", correlation_matrix)

#infant correlation
correlation_matrix = D_infant.drop(columns=['Sex']).corr()

# Print the correlation matrix
print("Correlation matrix for infant abalones:\n", correlation_matrix)

# Total correlation
correlation_matrix = D.drop(columns=["Sex"]).corr()

# Print the correlation matrix
print("Correlation matrix for all abalones:\n", correlation_matrix)


# Male covariance
covariance_matrix_male = D_male.drop(columns=['Sex']).cov()

# Print the covariance matrix
print("Covariance matrix for male abalones:\n", covariance_matrix_male)

# Female covariance
covariance_matrix_female = D_female.drop(columns=['Sex']).cov()

# Print the covariance matrix
print("Covariance matrix for female abalones:\n", covariance_matrix_female)

# Infant covariance
covariance_matrix_infant = D_infant.drop(columns=['Sex']).cov()

# Print the covariance matrix
print("Covariance matrix for infant abalones:\n", covariance_matrix_infant)

# Total covariance
covariance_matrix = D.drop(columns=["Sex"]).cov()

# Print the covariance matrix
print("Covariance matrix for all abalones:\n", covariance_matrix)