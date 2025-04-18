import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';')
D = pd.read_csv('abalone/abalone.data', sep=';')
# D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';') der hvor i har filen

N = len(D)

Male = D["Sex"]== "M"
Female = D["Sex"] == "F"
Infant = D["Sex"] == "I"

# "Sex";"Length";"Diameter";"Height";"Whole weight";"Shucked weight";"Viscera weight";"Shell weight";"Rings"
# parameters

# Assign values based on sex
D_male = D[Male]
D_female = D[Female]
D_infant = D[Infant]

D_clean = D.drop(columns=["Sex"])

#Stats about abalones
maleinfo = D_male.describe()
femaleinfo = D_female.describe()
infantinfo = D_infant.describe()
datainfo = D.describe()
# Convert datainfo to LaTeX format
datainfo_latex = datainfo.to_latex()

D['Sex'] = D['Sex'].map({'M': 1, 'F': 2, 'I': 3})
D_male['Sex'] = D_male['Sex'].map({'M': 1, 'F': 2, 'I': 3})
D_female['Sex'] = D_female['Sex'].map({'M': 1, 'F': 2, 'I': 3})
D_infant['Sex'] = D_infant['Sex'].map({'M': 1, 'F': 2, 'I': 3})

parameters = D.columns[1:]
C = len(D.columns) - 1 # amount of attributes (8)
classDict = dict(zip(parameters, range(C))) # dictionary of the parameters and their indices
print(classDict)

y = np.asarray([classDict[param] for param in parameters]) # indices of the parameters
print(y)


# train-test split
# Split the data into training and test sets (50% each)
Abalone_training = D.sample(frac=0.5, random_state=1) # 50% of the data for training
Abalone_test = D.drop(Abalone_training.index) # 50% of the data for testing
Abalone_training.to_csv('abalone_training.csv', sep=';', index=False)
Abalone_test.to_csv('abalone_test.csv', sep=';', index=False)
