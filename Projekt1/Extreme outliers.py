import pandas as pd
from data import D_male, D_female, D_infant, D_clean

# Ensure data is converted to Pandas Series if not already
D_male = pd.Series(D_male.values.flatten()) if not isinstance(D_male, pd.Series) and D_male.ndim > 1 else pd.Series(D_male) if not isinstance(D_male, pd.Series) else D_male
D_female = pd.Series(D_female.values.flatten()) if not isinstance(D_female, pd.Series) and D_female.ndim > 1 else pd.Series(D_female) if not isinstance(D_female, pd.Series) else D_female
D_infant = pd.Series(D_infant.values.flatten()) if not isinstance(D_infant, pd.Series) and D_infant.ndim > 1 else pd.Series(D_infant) if not isinstance(D_infant, pd.Series) else D_infant
D_clean = pd.Series(D_clean.values.flatten()) if not isinstance(D_clean, pd.Series) and D_clean.ndim > 1 else pd.Series(D_clean) if not isinstance(D_clean, pd.Series) else D_clean

def find_extreme_outliers(data):
    data = data.dropna()
    mean = data.mean()
    std = data.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

#D_male, D_female, D_infant, D_clean are already defined in the data.py file
# Finding extreme outliers for each group

male_outliers = find_extreme_outliers(D_male)
female_outliers = find_extreme_outliers(D_female)
infant_outliers = find_extreme_outliers(D_infant)
data_outliers = find_extreme_outliers(D_clean)

print("Male Outliers:\n", male_outliers)
print("Female Outliers:\n", female_outliers)
print("Infant Outliers:\n", infant_outliers)
print("Data Outliers:\n", data_outliers)