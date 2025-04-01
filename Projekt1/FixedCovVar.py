import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from data import *
import seaborn as sns
import pandas as pd

# Assuming D, D_male, D_female, D_infant are pandas DataFrames
# Encode 'Sex' column
D['Sex'] = D['Sex'].map({'M': 1, 'F': 2, 'I': 3})
D_male['Sex'] = D_male['Sex'].map({'M': 1, 'F': 2, 'I': 3})
D_female['Sex'] = D_female['Sex'].map({'M': 1, 'F': 2, 'I': 3})
D_infant['Sex'] = D_infant['Sex'].map({'M': 1, 'F': 2, 'I': 3})

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

variable_names = D.columns

# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(cov, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over covariance in abalone-dataset")
plt.savefig('heatmapcovall.png')
plt.show()

# Plot heatmap for D_male
plt.figure(figsize=(10, 8))
sns.heatmap(cov_male, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over covariance in abalone-dataset (Male)")
plt.show()

# Plot heatmap for D_female
plt.figure(figsize=(10, 8))
sns.heatmap(cov_female, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over covariance in abalone-dataset (Female)")
plt.show()

# Plot heatmap for D_infant
plt.figure(figsize=(10, 8))
sns.heatmap(cov_infant, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over covariance in abalone-dataset (Infant)")
plt.show()
