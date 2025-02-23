import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';')
# D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';') der hvor i har filen
# D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';') der hvor i har filen

Male = D["Sex"]== "M"
Female = D["Sex"] == "F"
Infant = D["Sex"] == "I"

# "Sex";"Length";"Diameter";"Height";"Whole weight";"Shucked weight";"Viscera weight";"Shell weight";"Rings"
# parameters


# Assign values based on sex
D_male = D[Male]
D_female = D[Female]
D_infant = D[Infant]

# Display the first few rows of each DataFrame
print("Male:\n", D_male.head())
print("Female:\n", D_female.head())
print("Infant:\n", D_infant.head())

#Stats about abalones
maleinfo = D_male.describe()
femaleinfo = D_female.describe()
infantinfo = D_infant.describe()
datainfo = D.describe()

# print("Male Info:\n", maleinfo)
# print("Female Info:\n", femaleinfo)
# print("Infant Info:\n", infantinfo)
# print("Data Info:\n", datainfo)

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