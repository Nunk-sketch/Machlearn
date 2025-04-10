import numpy as np
import pandas as pd
import numpy as np

#'M': 1, 'F': 2, 'I': 3

D = pd.read_csv('../abalone/abalone.data', sep=';')
D_test = pd.read_csv('../abalone/abalone_test.csv', sep=';')
D_train = pd.read_csv('../abalone/abalone_training.csv', sep=';')

#most common sex in the training set is the aim of this classification

sex_counts_test = D_test['Sex'].value_counts()
sex_counts_train = D_train["Sex"].value_counts()

# there are most males(1) in both sets, so the base case will be to make a model to predict males
# we will use the training set to train the model and the test set to test it


#abalone ages will be divided into three classes, based on rings: child = 0-6, adult = 7-15, old = 15<
def classify_age(rings):
    if rings <= 6:
        return 'child'
    elif 7 <= rings <= 15:
        return 'adult'
    else:
        return 'old'
# Apply the classification to the datasets
D['AgeClass'] = D['Rings'].apply(classify_age)
D_test['AgeClass'] = D_test['Rings'].apply(classify_age)
D_train['AgeClass'] = D_train['Rings'].apply(classify_age)

D['AgeClass'] = D['AgeClass'].map({'child': 0, 'adult': 1, 'old': 2})
D['Sex'] = D['Sex'].map({'M': 1, 'F': 2, 'I': 3})
D_test['AgeClass'] = D_test['AgeClass'].map({'child': 0, 'adult': 1, 'old': 2})
D_train['AgeClass'] = D_train['AgeClass'].map({'child': 0, 'adult': 1, 'old': 2})


x_clas = D.drop(columns=["Rings", "AgeClass"])
x_reg = D.drop(columns=['Rings', "AgeClass"])

y_clas = D['AgeClass']
y_reg = D['Rings']

# Separate the data into inputs (x_data) and output (y_data)
x_train_clas = (D_train.drop(columns=['Rings',"AgeClass"]))  # Drop the 'Sex' column to get the inputs
x_train_reg = D_train.drop(columns=['Rings', 'AgeClass'])
y_train_clas = (D_train['AgeClass'] ) # Use the 'Sex' column as the output
y_train_reg = D_train['Rings']  # Use the 'Rings' column as the output

x_test_clas = (D_test.drop(columns=["Rings","AgeClass"]))
x_test_reg = D_test.drop(columns=["Rings", "AgeClass"])
y_test_clas = (D_test["AgeClass"])
y_test_reg = D_test["Rings"]

# convert to feature matrix (classification)
x_clas_mat = x_clas.values
x_train_mat_clas = x_train_clas.values
x_test_mat_clas = x_test_clas.values

y_clas_mat = y_clas.values
y_train_mat_clas = y_train_clas.values
y_test_mat_clas = y_test_clas.values

# convert to feature matrix (regression)
x_mat_reg = x_reg.values
# Standardize the data
x_mat_reg = (x_mat_reg - x_mat_reg.mean(axis=0)) / x_mat_reg.std(axis=0)

y_mat_reg = y_reg.values
# Standardize the data
y_mat_reg = (y_mat_reg - y_mat_reg.mean(axis=0)) / y_mat_reg.std(axis=0)