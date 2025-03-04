import numpy as np
import matplotlib.pyplot as plt
from data import *
# Function to find extreme outliers
def find_extreme_outliers(data):
    data = data.dropna()
    mean = data.mean()
    std = data.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# Finding extreme outliers for each group
male_outliers = find_extreme_outliers(D_male.drop(columns=["Sex"]))
female_outliers = find_extreme_outliers(D_female.drop(columns=["Sex"]))
infant_outliers = find_extreme_outliers(D_infant.drop(columns=["Sex"]))
data_outliers = find_extreme_outliers(D_clean)

print("Male Outliers:\n", male_outliers)
print("Female Outliers:\n", female_outliers)
print("Infant Outliers:\n", infant_outliers)
print("Data Outliers:\n", data_outliers)