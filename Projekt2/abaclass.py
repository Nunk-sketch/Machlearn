import pandas as pd

#'M': 1, 'F': 2, 'I': 3

D = pd.read_csv('../abalone/abalone.data', sep=';')
D_test = pd.read_csv('../abalone/abalone_test.csv', sep=';')
D_train = pd.read_csv('../abalone/abalone_training.csv', sep=';')

attributeNames = D.columns.drop("Sex")

#most common sex in the training set is the aim of this classification

sex_counts_test = D_test['Sex'].value_counts()
sex_counts_train = D_train["Sex"].value_counts()

# there are most males(1) in both sets, so the base case will be to make a model to predict males
# we will use the training set to train the model and the test set to test it

x_clas = D.drop(columns=['Sex'])
x_reg = D.drop(columns=['Rings', "Sex"]) #TODO Rewrite sex as number

y_clas = D['Sex']
y_reg = D['Rings']

# Separate the data into inputs (x_data) and output (y_data)
x_train_clas = D_train.drop(columns=['Sex'])  # Drop the 'Sex' column to get the inputs
x_train_reg = D_train.drop(columns=['Rings', 'Sex'])
y_train_clas = D_train['Sex']  # Use the 'Sex' column as the output
y_train_reg = D_train['Rings']  # Use the 'Rings' column as the output

x_test_clas = D_test.drop(columns=["Sex"])
x_test_reg = D_test.drop(columns=["Rings", "Sex"])
y_test_clas = D_test["Sex"]
y_test_reg = D_test["Rings"]

# convert to feature matrix (classification)
x_train_mat_clas = x_train_clas.values
x_test_mat_clas = x_test_clas.values
y_train_mat_clas = y_train_clas.values
y_test_mat_clas = y_test_clas.values

# feature matrix
x_train_mat = x_train.values
x_test_mat = x_test.values
y_train_mat = y_train.values
y_test_mat = y_test.values
# convert to feature matrix (regression)
x_mat_reg = x_reg.values
x_train_mat_reg = x_train_reg.values
x_test_mat_reg = x_test_reg.values

y_mat_reg = y_reg.values
y_train_mat_reg = y_train_reg.values
y_test_mat_reg = y_test_reg.values

N_reg, M_reg = x_mat_reg.shape