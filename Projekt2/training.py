import torch
import torch.nn as nn
import torch.optim as optim
from networkModels import FCNN
from abaclass import *

# Initialize the model
model = FCNN()
#x_train, y_train, x_test, y_test
model.train_model(x_train_clas, y_train_clas, x_test_clas, y_test_clas, epochs=100)
# Evaluate the model on the test set
