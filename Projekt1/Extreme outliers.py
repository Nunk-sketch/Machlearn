import pandas as pd
from data import D_male, D_female, D_infant, D_clean

# Ensure data is converted to Pandas Series if not already
D_male = pd.Series(D_male.values.flatten()) if not isinstance(D_male, pd.Series) and D_male.ndim > 1 else pd.Series(D_male) if not isinstance(D_male, pd.Series) else D_male
D_female = pd.Series(D_female.values.flatten()) if not isinstance(D_female, pd.Series) and D_female.ndim > 1 else pd.Series(D_female) if not isinstance(D_female, pd.Series) else D_female
D_infant = pd.Series(D_infant.values.flatten()) if not isinstance(D_infant, pd.Series) and D_infant.ndim > 1 else pd.Series(D_infant) if not isinstance(D_infant, pd.Series) else D_infant
D_clean = pd.DataFrame(D_clean) if not isinstance(D_clean, pd.DataFrame) else D_clean

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
# Assuming the data contains a column 'rings' for abalones
def find_extremes_for_rings(data):
    if 'Rings' not in data.columns:
        raise ValueError("The dataset does not contain a 'rings' column.")
    rings_data = data['Rings']
    return find_extreme_outliers(rings_data)

# Example usage (assuming D_clean is a DataFrame with a 'rings' column)
abalone_rings_outliers = find_extremes_for_rings(D_clean)
print("Abalone Rings Outliers:\n", abalone_rings_outliers)
if not abalone_rings_outliers.empty:
    print("Highest value for 'Rings' outlier:", abalone_rings_outliers.max())
else:
    print("No outliers found for 'Rings'.")