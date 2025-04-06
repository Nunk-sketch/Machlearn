import numpy as np
import matplotlib.pyplot as plt
from data import *
import seaborn as sns

# Drop the "Sex" column
D_male = D_male.drop(columns=["Sex"])
D_female = D_female.drop(columns=["Sex"])
D_infant = D_infant.drop(columns=["Sex"])

# Calculate correlation matrices
corr = D_clean.corr()
corr_male = D_male.corr()
corr_female = D_female.corr()
corr_infant = D_infant.corr()

variable_names = D_clean.columns

# Plot heatmap for D_clean
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over correlation i abalone-datasættet")
plt.savefig('heatmapcorrall.png')
plt.show()

# Plot heatmap for D_male
plt.figure(figsize=(10, 8))
sns.heatmap(corr_male, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over correlation i abalone-datasættet (Male)")
plt.savefig('heatmapcorr_male.png')
plt.show()

# Plot heatmap for D_female
plt.figure(figsize=(10, 8))
sns.heatmap(corr_female, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over correlation i abalone-datasættet (Female)")
plt.savefig('heatmapcorr_female.png')
plt.show()

# Plot heatmap for D_infant
plt.figure(figsize=(10, 8))
sns.heatmap(corr_infant, annot=False, cmap='coolwarm', linewidths=0.5, xticklabels=variable_names, yticklabels=variable_names)
plt.title("Heatmap over correlation i abalone-datasættet (Infant)")
plt.savefig('heatmapcorr_infant.png')
plt.show()
