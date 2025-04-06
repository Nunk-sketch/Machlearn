import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from data import *
import seaborn as sns

D_male = D_male.drop(columns=["Sex"])
D_female = D_female.drop(columns=["Sex"])
D_infant = D_infant.drop(columns=["Sex"])


mean = np.mean(D_clean, axis=0)
std = np.std(D_clean, axis=0)
D_standardized =  (D_clean - mean) / std

cov = np.cov(D_standardized, rowvar=False)

variable_names = D_clean.columns

# Lav et heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cov, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)

plt.title("Heatmap over covariance in abalone-dataset")
plt.savefig('heatmapcovall.png')

# Vis plot
plt.show()


# Standardize D_male
mean_male = np.mean(D_male, axis=0)
std_male = np.std(D_male, axis=0)
D_male_standardized = (D_male - mean_male) / std_male

# Standardize D_female
mean_female = np.mean(D_female, axis=0)
std_female = np.std(D_female, axis=0)
D_female_standardized = (D_female - mean_female) / std_female

# Standardize D_infant
mean_infant = np.mean(D_infant, axis=0)
std_infant = np.std(D_infant, axis=0)
D_infant_standardized = (D_infant - mean_infant) / std_infant

# Calculate covariance matrices
cov_male = np.cov(D_male_standardized, rowvar=False)
cov_female = np.cov(D_female_standardized, rowvar=False)
cov_infant = np.cov(D_infant_standardized, rowvar=False)

# Plot heatmap for D_male
plt.figure(figsize=(10, 8))
sns.heatmap(cov_male, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over covariance in abalone-dataset (Male)")
plt.savefig('heatmapcov_male.png')
plt.show()

# Plot heatmap for D_female
plt.figure(figsize=(10, 8))
sns.heatmap(cov_female, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over covariance in abalone-dataset (Female)")
plt.savefig('heatmapcov_female.png')
plt.show()

# Plot heatmap for D_infant
plt.figure(figsize=(10, 8))
sns.heatmap(cov_infant, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over covariance in abalone-dataset (Infant)")
plt.savefig('heatmapcov_infant.png')
plt.show()