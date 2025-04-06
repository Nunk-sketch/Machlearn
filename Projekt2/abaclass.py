import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#'M': 1, 'F': 2, 'I': 3

D_test = pd.read_csv('abalone/abalone_test.csv', sep=';')
D_train = pd.read_csv('abalone/abalone_training.csv', sep=';')

#most common sex in the training set is the aim of this classification


sex_counts_test = D_test['Sex'].value_counts()
sex_counts_train = D_train["Sex"].value_counts()

# there are most males(1) in both sets, so the base case will be to make a model to predict males
# we will use the training set to train the model and the test set to test it


# Separate the data into inputs (x_data) and output (y_data)
x_train = D_train.drop(columns=['Sex'])  # input
y_train = D_train['Sex']  #output

x_test = D_test.drop(columns=["Sex"])
y_test = D_test["Sex"]

# feature matrix
x_train_mat = x_train.values
x_test_mat = x_test.values
y_train_mat = y_train.values
y_test_mat = y_test.values
